if [ -z ${IP+x} ]; then echo "IP is unset"; else
scp felipemoran@$IP:~/a4_sobel/out/* ./out/
fi;
