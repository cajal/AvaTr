wget https://www.dropbox.com/s/jg2og2ogf0l7ke9/Avatar10Mix2.tar.gz?dl=0
mv Avatar10Mix2.tar.gz?dl=0 Avatar10Mix2.tar.gz
tar zxvf Avatar10Mix2.tar.gz

# pre-processing
cd Avatar10Mix2/metadata/
find . -iname "mixture*" | wc

for filename in `find . -iname "mixture*"`
do
    sed -i 's/Avatar10Mix2\///g' $filename
done

cd -
