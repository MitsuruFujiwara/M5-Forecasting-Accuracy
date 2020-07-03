# anacondaなどのインストール
sudo apt update
sudo apt upgrade
sudo apt-get install -y wget
sudo apt-get install -y unzip
wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh
sh Anaconda3-2019.10-Linux-x86_64.sh
rm Anaconda3-2019.10-Linux-x86_64.sh

# ここで一旦sshのウインドウを閉じる
# ブラウザでsshウインドウを再起動後、追加で必要なパッケージなどをインストール
conda install -c conda-forge feather-format
conda install -c conda-forge lightgbm
pip install kaggle
pip install holidays

# kaggle API用のjsonファイルを所定の位置に移動（設定ボタン→ファイルをアップロード）
mv kaggle.json /home/fujiwara52jp/.kaggle
chmod 600 /home/fujiwara52jp/.kaggle/kaggle.json

# githubのリポジトリをクローン
git clone -b late_submission https://github.com/MitsuruFujiwara/M5-Forecasting-Accuracy.git
cd M5-Forecasting-Accuracy/
sh init.sh

# LINE通知用のトークンを所定の位置に移動（任意）
cd ..
mv line_token.txt M5-Forecasting-Accuracy/input/

git push origin late_submission
