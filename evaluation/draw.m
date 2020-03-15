x=0:0.2:1;
r = [38.6, 40.5, 38.0, 40.7, 42.2, 41.0];
bl = [31.8, 31.8, 31.8, 31.8, 31.8, 31.8];
plot(x,r,'-*b',x,bl,'->r');
axis([0,1,31,44])
set(gca, 'xtick', [0:0.2:1]);
set(gca, 'ytick', [31:1.0:44]);

legend('TAAN','baseline')
xlabel('\lambda_{IR}')
ylabel('Rank-1(%)')