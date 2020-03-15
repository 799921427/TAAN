x=0:0.2:1;
map = [40.0, 41.5, 42.9, 41.4, 43.0, 40.7];
bl = [36.1, 36.1, 36.1, 36.1, 36.1, 36.1];
plot(x,map,'-*b',x,bl,'->r');
axis([0,1,36,44])
set(gca, 'xtick', [0:0.2:1]);
set(gca, 'ytick', [35:1.0:44]);

legend('TAAN','baseline')
xlabel('\lambda_{IR}')
ylabel('mAP(%)')