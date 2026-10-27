param(
    [string]$PdfDirectory = (Join-Path $PSScriptRoot "..\data\raw\all_pdfs"),
    [string]$TextDirectory = (Join-Path $PSScriptRoot "..\data\raw\all_pdfs_text"),
    [string]$ManifestPath = (Join-Path $PSScriptRoot "..\data\processed_graph\pdf_manifest.json")
)

$ErrorActionPreference = "Stop"
$PdfDirectory = [System.IO.Path]::GetFullPath($PdfDirectory)
$TextDirectory = [System.IO.Path]::GetFullPath($TextDirectory)
$ManifestPath = [System.IO.Path]::GetFullPath($ManifestPath)
if (-not (Test-Path -LiteralPath $PdfDirectory -PathType Container)) {
    throw "PDF directory not found: $PdfDirectory"
}

[void][System.Reflection.Assembly]::LoadWithPartialName("System.Runtime.WindowsRuntime")
$runtimeAssembly = [System.Reflection.Assembly]::LoadWithPartialName("System.Runtime.WindowsRuntime")
$extensionType = $runtimeAssembly.GetType("System.WindowsRuntimeSystemExtensions")
$asTaskOperation = $extensionType.GetMethods([System.Reflection.BindingFlags]"Public,Static") |
    Where-Object {
        $_.Name -eq "AsTask" -and $_.IsGenericMethodDefinition -and
        $_.GetGenericArguments().Length -eq 1 -and $_.GetParameters().Length -eq 1
    } | Select-Object -First 1
$asTaskAction = $extensionType.GetMethods([System.Reflection.BindingFlags]"Public,Static") |
    Where-Object { $_.Name -eq "AsTask" -and -not $_.IsGenericMethod -and $_.GetParameters().Length -eq 1 } |
    Select-Object -First 1

function Wait-WinRtOperation($Operation, [Type]$ResultType) {
    $task = $script:asTaskOperation.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

function Wait-WinRtAction($Action) {
    $task = $script:asTaskAction.Invoke($null, @($Action))
    $task.Wait()
}

$storageFileType = [Type]::GetType("Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime")
$pdfDocumentType = [Type]::GetType("Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime")
$streamType = [Type]::GetType("Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime")
$decoderType = [Type]::GetType("Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime")
$renderOptionsType = [Type]::GetType("Windows.Data.Pdf.PdfPageRenderOptions, Windows.Data.Pdf, ContentType=WindowsRuntime")
$softwareBitmapType = [Type]::GetType("Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime")
$ocrEngineType = [Type]::GetType("Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime")
$ocrResultType = [Type]::GetType("Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType=WindowsRuntime")
foreach ($requiredType in @($storageFileType, $pdfDocumentType, $streamType, $decoderType, $renderOptionsType, $softwareBitmapType, $ocrEngineType, $ocrResultType)) {
    if ($null -eq $requiredType) { throw "A required Windows PDF/OCR runtime type is unavailable." }
}

$getFileMethod = $storageFileType.GetMethods([System.Reflection.BindingFlags]"Public,Static") |
    Where-Object { $_.Name -eq "GetFileFromPathAsync" -and $_.GetParameters().Length -eq 1 } |
    Select-Object -First 1
$loadPdfMethod = $pdfDocumentType.GetMethods([System.Reflection.BindingFlags]"Public,Static") |
    Where-Object { $_.Name -eq "LoadFromFileAsync" -and $_.GetParameters().Length -eq 1 } |
    Select-Object -First 1
$createDecoderMethod = $decoderType.GetMethods([System.Reflection.BindingFlags]"Public,Static") |
    Where-Object { $_.Name -eq "CreateAsync" -and $_.GetParameters().Length -eq 1 } |
    Select-Object -First 1
$recognizeMethod = $ocrEngineType.GetMethods([System.Reflection.BindingFlags]"Public,Instance") |
    Where-Object { $_.Name -eq "RecognizeAsync" -and $_.GetParameters().Length -eq 1 } |
    Select-Object -First 1
$ocrEngine = $ocrEngineType.GetMethod("TryCreateFromUserProfileLanguages").Invoke($null, @())
if ($null -eq $ocrEngine) {
    throw "Windows OCR has no language pack enabled for the current user. Add an English OCR language pack and rerun."
}

New-Item -ItemType Directory -Force -Path $TextDirectory | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ManifestPath) | Out-Null
$pdfs = @(Get-ChildItem -LiteralPath $PdfDirectory -Filter "*.pdf" -File | Sort-Object Name)
if ($pdfs.Count -eq 0) { throw "No PDFs found in $PdfDirectory" }
$manifest = [System.Collections.Generic.List[object]]::new()
$pdfIndex = 0
foreach ($pdf in $pdfs) {
    $pdfIndex++
    $warnings = [System.Collections.Generic.List[string]]::new()
    $text = [System.Text.StringBuilder]::new()
    $pageCount = 0
    try {
        $storageFile = Wait-WinRtOperation ($getFileMethod.Invoke($null, @($pdf.FullName))) $storageFileType
        $document = Wait-WinRtOperation ($loadPdfMethod.Invoke($null, @($storageFile))) $pdfDocumentType
        $pageCount = [int]$document.PageCount
        for ($pageIndex = 0; $pageIndex -lt $pageCount; $pageIndex++) {
            try {
                $page = $document.GetPage([uint32]$pageIndex)
                $stream = [Activator]::CreateInstance($streamType)
                $renderOptions = [Activator]::CreateInstance($renderOptionsType)
                $renderOptions.DestinationWidth = [uint32][Math]::Ceiling($page.Size.Width * 2)
                $renderOptions.DestinationHeight = [uint32][Math]::Ceiling($page.Size.Height * 2)
                Wait-WinRtAction ($page.RenderToStreamAsync($stream, $renderOptions))
                $decoder = Wait-WinRtOperation ($createDecoderMethod.Invoke($null, @($stream))) $decoderType
                $bitmap = Wait-WinRtOperation ($decoder.GetSoftwareBitmapAsync()) $softwareBitmapType
                $result = Wait-WinRtOperation ($recognizeMethod.Invoke($ocrEngine, @($bitmap))) $ocrResultType
                [void]$text.AppendLine("[Page $($pageIndex + 1)]")
                [void]$text.AppendLine($result.Text)
                [void]$text.AppendLine()
            }
            catch {
                $warnings.Add("Page $($pageIndex + 1): $($_.Exception.Message)")
            }
        }
    }
    catch {
        $warnings.Add("PDF open or parse: $($_.Exception.Message)")
    }

    $textPath = Join-Path $TextDirectory ($pdf.BaseName + ".txt")
    $content = $text.ToString().Trim()
    [System.IO.File]::WriteAllText($textPath, $content, [System.Text.UTF8Encoding]::new($false))
    $manifest.Add([pscustomobject]@{
        file = $pdf.Name
        source_path = $pdf.FullName
        extracted_text = $textPath
        pages = $pageCount
        characters = $content.Length
        extraction_method = "Windows PDF renderer + Windows OCR"
        warnings = @($warnings)
    })
    Write-Host ("[{0}/{1}] {2}: {3} pages, {4} characters{5}" -f $pdfIndex, $pdfs.Count, $pdf.Name, $pageCount, $content.Length, $(if ($warnings.Count) { ", $($warnings.Count) warning(s)" } else { "" }))
}

[System.IO.File]::WriteAllText(
    $ManifestPath,
    ($manifest | ConvertTo-Json -Depth 5),
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "Manifest saved to $ManifestPath"
