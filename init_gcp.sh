apt-get install -y wget unzip
wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh
sh Anaconda3-2019.10-Linux-x86_64.sh

pip install kaggle
mv kaggle.json /home/fujiwara52jp/.kaggle
chmod 600 /home/fujiwara52jp/.kaggle/kaggle.json

conda install -c conda-forge feather-format
conda install -c conda-forge lightgbm

git clone https://github.com/MitsuruFujiwara/M5-Forecasting-Accuracy.git
cd M5-Forecasting-Accuracy/
sh init.sh
mv line_token.txt M5-Forecasting-Accuracy/input/
