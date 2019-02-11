if [ -z ${IP+x} ]; then echo "IP is unset"; else
scp -r ./* felipemoran@$IP:~/a4_sobel
fi
