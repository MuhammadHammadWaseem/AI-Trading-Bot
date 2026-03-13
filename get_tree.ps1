function Get-Tree {
    param (
        [string]$Path = '.',
        [string]$Prefix = '',
        [string[]]$Exclude = @('.git', 'venv', '.venv', '__pycache__', 'node_modules', 'saved_models', '.pytest_cache')
    )

    try {
        $items = Get-ChildItem -Path $Path -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -notin $Exclude } | Sort-Object Name
    } catch {
        return
    }

    $count = $items.Count
    $i = 0

    foreach ($item in $items) {
        $i++
        $isLast = $i -eq $count
        $branch = if ($isLast) { "└── " } else { "├── " }
        
        Write-Output ($Prefix + $branch + $item.Name)

        if ($item.PSIsContainer) {
            $extension = if ($isLast) { "    " } else { "│   " }
            Get-Tree -Path $item.FullName -Prefix ($Prefix + $extension) -Exclude $Exclude
        }
    }
}

"AI Trading Bot" | Out-File -FilePath "project_structure.txt" -Encoding utf8
Get-Tree | Out-File -FilePath "project_structure.txt" -Encoding utf8 -Append
