x=0:0.2:1;
map = [37.6, 38.2, 38.8, 40.5, 41.6, 43.0];
bl = [36.1, 36.1, 36.1, 36.1, 36.1, 36.1];
plot(x,r,'-*b',x,bl,'->r');
axis([0,1,36,44])
set(gca, 'xtick', [0:0.2:1]);
set(gca, 'ytick', [36:1.0:44]);

legend('TAAN','baseline')
xlabel('\lambda_{RGB}')
ylabel('mAP(%)')