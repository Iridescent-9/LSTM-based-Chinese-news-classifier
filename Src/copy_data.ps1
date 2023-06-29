$MAXCOUNT = 6500
$categories = @("财经", "房产", "家居", "教育", "科技", "时尚", "时政", "体育", "游戏", "娱乐")

foreach ($category in $categories)
{
    $dir     = "D:\MyCode\Graduation project\THUCNews\THUCNews\$category"
    $newdir  = "D:\MyCode\Graduation project\Data\$category"
    $COUNTER = 1
    $files   = Get-ChildItem -File -Path $dir
    if (Test-Path $newdir)
    {
        Remove-Item -Recurse -Force $newdir
    }
    else
    {
        New-Item -ItemType Directory -Path $newdir | Out-Null
    }
    foreach ($file in $files)
    {
        Copy-Item -Path $file.FullName -Destination $newdir
        if ($COUNTER -ge $MAXCOUNT)
        {
            Write-Host "finished item:$category"
            break
        }
        $COUNTER++
    }
}
