1. Python 가상 환경 구축
   a. Miniconda 설치
     a-1 Anaconda prompt 실행  -> 가상 환경 확인 (conda info --envs)
     b-2 Python 설치
         $conda search python -> $conda create --name (가상환경 이름(Gen)) python=3.11.9
         $conda info --envs   => (Gen) 확인
         $conda activate (Gen) 
   b. 필요 libarary 설치 (@가상환경)
      (Gen) C:\...> $ pip install jupyter
      (Gen) C:\...> $ pin install streamlit

2. Vscode 에서 Jupyter 환경 사용
   a. Python, Python Debugger, Pylance, Jyupyter extention 설치
   b. (Ctrl + Shift + P) -> Python:Select Interpreter -> Python 3.11.9(Gen) 

3. UI 실행 (Vscode 기준) 
   a. git bash 창에서, (Gen) 접속 
   (Gen) C:\..> $streamlit run main.py 
