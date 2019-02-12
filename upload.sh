if [ -z ${IP+x} ]; then 
	echo "IP is unset"; 
else
	# scp -r include/* felipemoran@$IP:~/a4_sobel/include
	# scp -r src/* felipemoran@$IP:~/a4_sobel/src
	# scp -r main.cpp felipemoran@$IP:~/a4_sobel
	rsync -ru ./ felipemoran@$IP:~/a4_sobel
fi
