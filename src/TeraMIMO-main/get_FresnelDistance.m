function d_fres = get_FresnelDistance(p)

pos_TxAE1 = get_AEPosition(p,1,1,1,1,'T','G');
pos_TxAEn = get_AEPosition(p,p.Mt,p.Nt,p.Mat,p.Nat,'T','G');
ApertureTx = norm(abs(pos_TxAEn - pos_TxAE1));

pos_RxAE1 = get_AEPosition(p,1,1,1,1,'R','G');
pos_RxAEn = get_AEPosition(p,p.Mr,p.Nr,p.Mar,p.Nar,'R','G');
ApertureRx = norm(abs(pos_RxAEn - pos_RxAE1));

lambda = p.c / p.Fc;
d_fres = 0.62*sqrt((ApertureRx+ApertureTx)^3/lambda);