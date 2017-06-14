set terminal 'png'
set title 'comparison'
set datafile separator " " 
set output 'time.png'
set ylabel 'time(sec)'
plot "time.txt" using 1:2 with lines title 'naive',\
'' using 1:3 with lines title 'sse',\
'' using 1:4 with lines title 'sse prefetch',\
'' using 1:5 with lines title 'avx',\
'' using 1:6 with lines title 'avx prefetch'
