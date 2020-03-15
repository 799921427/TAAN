x=0:0.2:1;
r = [37.3, 37.8, 38.4, 39.4, 41.2, 42.2];
bl = [31.8, 31.8, 31.8, 31.8, 31.8, 31.8];
plot(x,r,'-*b',x,bl,'->r');
axis([0,1,31,44])
set(gca, 'xtick', [0:0.2:1]);
set(gca, 'ytick', [31:1.0:44]);

legend('TAAN','baseline')
xlabel('\lambda_{RGB}')
ylabel('Rank-1(%)')