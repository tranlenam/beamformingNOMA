clear all
clc
rng('default');

nUsers = 3;
nTx = 8;
f = 4000; %Signal frequency in MHz
Ptx = 20; %Total transmit power in Watts
Bw = 5; %Signal bandwidth in MHz
a = 4; b = 6.5e-3; c = 17.1; %Terrain constants from SUI model
s = 9.4; %Shadowing effect in dB, 8.2dB < s < 10.6dB
hb = 30; %Base station height in m, 15m < hb < 40m

d0 = 100; %Reference distance in m
k = physconst('Boltzmann');

g = a - b*hb + c/hb;
lambda = 300/f;
A = 20*log10(4*pi*d0/lambda);
Lbf = 6*log10(f/2000);
noisepower = k*290*(Bw*1e6)/2; %Thermal noise power
PL = @(d) 10^((A + s + Lbf + 10*g*log10(d/d0))/10);
userDistances = [1000 700 400];
userPathLoss = arrayfun(PL, userDistances(1:nUsers));

myalpha = 16;
maxIteration  = 20;
tol = 1e-4;
h = zeros(nTx, nUsers) ;
w = sdpvar(nTx, nUsers, 'full', 'complex');
mygamma = sdpvar(nUsers, 1);
options = sdpsettings('verbose',0,'solver','mosek');

smallScaleFading = (randn(nTx, nUsers) + 1i*randn(nTx, nUsers))/sqrt(2);
channelNorms = zeros(nUsers,1);
for iUser = 1 : nUsers
    h(:, iUser) = smallScaleFading(:, iUser)*(userPathLoss(iUser)^(-1/2));
    channelNorms(iUser) = norm(h(:, iUser),2);
end

[sorted, sortOrder] = sort(channelNorms);
h = h(:, sortOrder);

%Normalise
h = h/sqrt(noisepower); % normalise the channel with noise power for better numerical stability
effnoisepower=1; % set the effective noise power to 1 due to the step above

w0 = sqrt(Ptx/(nUsers*nTx))*ones(nTx, nUsers);
[sumrate,gamma0] = ComputeSumRate(h,w0,effnoisepower);
converengeflag =0;
solver_time = 0;
objProgress = zeros(maxIteration,1);
sumRateProgressAlg2adaptive=zeros(maxIteration,1);
for iIter = 1:maxIteration
    obj = 0;
    for  iUser = 1:nUsers
        mybeta = myalpha*(1+gamma0(iUser))^2;
        qadaptive = 1/(2 + mybeta/iIter);
        
        obj = obj+log(1+gamma0(iUser))+1/(1+gamma0(iUser))...
            *(mygamma(iUser)-gamma0(iUser))...
            -qadaptive*(mygamma(iUser)-gamma0(iUser))^2;
    end
    
    %F = [w(:)'*w(:) <= Ptx]; % % sum power constraint
    F = [cone(w(:),sqrt(Ptx))]; % sum power constraint
    F = [F,mygamma(:) >=0];
    % (10c)
    for iUser = 1:nUsers
        for jUser = 1:nUsers-1
            %{
            F = [F,InnerProductApprox(h(:,iUser),w(:,jUser),w0(:,jUser)) ...
                >= abs(h(:,iUser)'*w(:,jUser+1))^2];
            %}
            t = InnerProductApprox(h(:,iUser),w(:,jUser),w0(:,jUser));
            F = [F,cone([h(:,iUser)'*w(:,jUser+1);(t-1)/2],...
                (t+1)/2)]; % this is equivalent to t >= |h(:,iUser)'*w(:,jUser+1)|^2
            
        end
    end
    % (11c)
    for iUser = 1:nUsers-1
        for jUser =iUser:nUsers
            interference = h(:,jUser)'*w(:,iUser+1:nUsers);
            %{
            F = [F,QuadraticOverLinearApprox(h(:,jUser),w(:,iUser),w0(:,iUser),mygamma(iUser),gamma0(iUser))...
                >=(effnoisepower+sum(abs(interference).^2))];
            %}
            t = QuadraticOverLinearApprox(h(:,jUser),w(:,iUser),...
                w0(:,iUser),mygamma(iUser),gamma0(iUser));
            F = [F,cone([sqrt(effnoisepower);interference(:);(t-1)/2],(t+1)/2)];
        end
    end
    
    F = [F,InnerProductApprox(h(:,nUsers),w(:,nUsers),w0(:,nUsers))...
         >= effnoisepower*mygamma(nUsers)];
    
    diagnose = optimize(F, -obj, options); % solve the problem
    
    if (diagnose.problem==0 )
        w0 = value(w);
        gamma0 = value(mygamma);
        objProgress(iIter) = sum(log(1+gamma0));
        sumRateProgressAlg2adaptive(iIter) = ComputeSumRate(h,w0,effnoisepower);
    else
        yalmiperror(diagnose.problem)
        break;
    end

    
end

plot(sumRateProgressAlg2adaptive, 'r')
hold on
plot(objProgress,'b')
legend('Sum rate ','Objective Sequence','location','southeast')
xlabel('Iteration count, $n$ ', 'Interpreter','latex')
ylabel('Sum Rate')

function [linearapprox] = QuadraticOverLinearApprox(x,y,y0,z,z0)
% This function returns the first order approximation to
% the quadratic over linear function in the form |x^{H}y|^2/z around the point y0,z0  
% See RHS of (9)
linearapprox = -abs(x'*y0)^2/z0 + 2*real(y0'*(x*x')*y)/z0...
    -abs(x'*y0)^2/(z0^2)*(z-z0);
end

function [z] = InnerProductApprox(x,y,y0)
%InnerProductApp: This function returns the first order approximation to
% the term |x^{H}y|^2 around the point y0,  
% See RHS of (8)
z = -abs(x'*y0)^2 + 2*real(y0'*(x*x')*y);
end

function [SumRate,effSINRs]= ComputeSumRate(h,w,Pn)
[~,nUsers]=size(h);
effSINRs = zeros(nUsers,1);
for iUser = 1:nUsers-1
    SINR = zeros(nUsers-iUser+1,1);
    for jUser =iUser:nUsers
        interference = h(:,jUser)'*w(:,iUser+1:nUsers);
        SINR(jUser-iUser+1) = abs(h(:,jUser)'*w(:,iUser))^2/...
            (Pn+norm(interference)^2);
    end
    effSINRs(iUser) = min(SINR);
end
effSINRs(nUsers) = abs(h(:,nUsers)'*w(:,nUsers))^2/Pn;

SumRate = sum(log(1+effSINRs));
end
