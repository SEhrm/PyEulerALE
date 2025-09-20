set terminal png giant size 700,500 fontscale 2 background rgb 'black'
set xlabel tc "white"
set ylabel tc "white"
set title tc "white"
set key tc "white"
set tics tc "white"
set border lc "white"

set output 'cp.png'
set datafile separator ","
set yrange reverse
set xrange [0 : 1]
set grid
set title "NACA-0012 at Mach number 0.5 and effective angle-of-attack 1.25 deg"
set xlabel "x / chord"
set ylabel "surface pressure coefficient"
plot 'cp_xfoil.csv' u 1:3 with linespoints pt 2 lc "web-blue" title "XFOIL", \
    'cp.csv' u 1:3 with points pt 13 lc "red" title "PyEulerALE", \

