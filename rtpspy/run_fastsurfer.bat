:: Activate fastsurfer-gpu env
call %HOMEPATH%\Anaconda3\condabin\activate fastsurfer-gpu

:: Check ig GPU is available
FOR /F "tokens=* USEBACKQ" %%F IN (`python -c "import torch; print(int(torch.cuda.is_available()))"`) DO (
	set gpu=%%F
)

:: Set batch size
if %gpu%==1 (
	:: GPU is vailable
	FOR /F "tokens=* USEBACKQ" %%F IN (`python -c "import torch; print(int((torch.cuda.get_device_properties(0).total_memory-1e9) // 500000000))"`) DO (
		set bsize=%%F
	)
) else (
	:: No GPU
	set /A bsize=8
)

cd c:\FastSurfer
python c:\FastSurfer\FastSurferCNN\eval.py --in_name %1 --out_name %2 --batch_size %bsize% --simple_run

