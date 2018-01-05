mycurl(){
        START=$(date +%s.%N)
        curl -i "http://localhost:8000/?image=http://cdn.mantii.com/photos/tumblr_o7qmrwopPC1ujwj6mo1_1280.jpg"
        #curl -i 10.0.135.197:8000/models/images/classification/classify_one.json -XPOST -F job_id=20151216-171530-b733 -F image_url=http://media1.santabanta.com/full1/Miscellaneous/Cartoon%20Characters/cartoon-characters-76a.jpg
        END=$(date +%s.%N)
        DIFF=$(echo "$END - $START" | bc)
}


export -f mycurl


while true;
do
        seq 10 | parallel -j4 mycurl
done
