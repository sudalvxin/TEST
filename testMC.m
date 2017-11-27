
%% A function for RPCA
function Out = testMC()
% mxn data matrix M with rank r
m = 500;
n = 500;
r = 5;

U0 = rand(m,r);
V0 = rand(r,n);

W = double(rand(m,n) > 0.5);
Omega = find(W);

%ground truth 
M0 = U0*V0; %+ones(m,n);
data = M0(Omega);
%adding Gaussian noise
Gn = rand(m,n)*0;
M = M0 + Gn;


% sparse outliers
G = double(rand(m,n) > 1);

% sparse outlier ratio
outlier_ratio = sum(sum(G))/(m*n);

% adding outliers with fixed magnatude of 1
M = M + G*1;
% adding column outiers
ratio = 0.7;
CoutN = round(n*ratio);
Omat = randn(m,CoutN)*2;
M(:,end - CoutN + 1:end) = M(:,end - CoutN + 1:end) + Omat;
%% Test algorithm
do_RegL1_ALM = 1;
do_SRMCCS = 0;
do_ORMC = 1;  % our method
do_GRASTA = 0;
do_ISVD = 0;
do_Schattenlp = 0;
%% do RegL1_ALM
if do_RegL1_ALM
    addpath 'G:\TestRPCA\MyMC\RegL1_ALM'
tic
[M_est U_est V_est L1_error] = RobustApproximation_M_UV_TraceNormReg(M,W,r,1e-3,1.2,1,0);
toc
%M_est(Omega) = data;
E = M0 - M_est;
E(:,end - CoutN + 1:end) = [];
RE = norm(E,'fro')/norm(M0(:,1:end-CoutN),'fro');
Out.RegL1 = RE;

rmpath 'G:\TestRPCA\MyMC\RegL1_ALM'
end
%% test IALM
if do_ISVD ==1;
   addpath 'G:\TestRPCA\MyMC\OnlineRPCA'
tic
%M_est = testORPCA(M,W,r);
[M_est,N,U] = NORPCA(M,W,r,10);
Out.observed = N;
toc
E = M0 - M_est;
E(:,end - CoutN + 1:end) = [];
RE = norm(E,'fro')/norm(M0(:,1:end - CoutN),'fro');
Out.OPRMC = RE;

rmpath 'G:\TestRPCA\MyMC\OnlineRPCA'
end
%% test GRASTA
if do_GRASTA ==1;
addpath 'G:\TestRPCA\MyMC\GRASTA_new'
tic
M_est = testGRASTA(M,Omega,r);
toc
E = M0 - M_est;
E(:,end - CoutN + 1:end) = [];
RE = norm(E,'fro')/norm(M0(:,1:end - CoutN),'fro');
Out.GRASTA = RE;
rmpath 'G:\TestRPCA\MyMC\GRASTA_new'
end
%% test ORMC
if do_ORMC ==1;
   addpath 'G:\TestRPCA\MyMC\m_ORMC'
tic
out = ORMC(M.*W,W,r,100);
a = out.re;
weighted = out.weight;
%plot
% plot(1:length(a),a);
plotweighted(weighted)

M_est = out.matrix;
toc
E = M0 - M_est;
E(:,end - CoutN + 1:end) = [];
RE = norm(E,'fro')/norm(M0(:,1:end - CoutN),'fro');
Out.ORMC = RE;
rmpath 'G:\TestRPCA\MyMC\m_ORMC'
end
%% test Lp
if do_Schattenlp == 1;
addpath 'G:\TestRPCA\MyMC\Schattenlp'
tic
[M_est ,~,~]=OtraceEEC_my(M,W, 1); % noiseless
%[M_est ,~,~]=OtraceEIC_my(M,W, 1,1e6); % noisy
[~,S,~] = svd(M_est,'econ');
toc
E = M0 - M_est;
E(:,end - CoutN + 1:end) = [];
RE = norm(E,'fro')/norm(M0(:,1:end - CoutN),'fro');
Out.Schattenlp = RE;
Out.rank = rank(M_est);
Out.S = S;
rmpath 'G:\TestRPCA\MyMC\Schattenlp'
end













