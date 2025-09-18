set terminal png small size 800,800
set output 'cp.png'
set datafile separator ","
set yrange reverse
set xrange [0 : 1]
set grid
set xlabel "x / chord"
set ylabel "cp"
plot 'cp.csv' using 1:3 with linespoints pointtype 2 linecolor "black" notitle
