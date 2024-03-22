echo "Dowmloading VGGish model and PCA parameters"
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

echo "Downloading the VGGish model code"
git clone https://github.com/tensorflow/models.git
cp models/research/audioset/vggish/* .
rm -rf models/


