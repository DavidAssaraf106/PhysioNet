function [max_freq] = f_wave_detection(ecg_lead,qrs,fs)
try
    % == prefiltering
    ecg = horzcat(ecg_lead{:});
    qrs = horzcat(qrs{:});
    LOW_CUT_FREQ = 4;
    HIGH_CUT_FREQ = 45;
    [b_lp,a_lp] = butter(5,HIGH_CUT_FREQ/(fs/2),'high');
    [b_bas,a_bas] = butter(2,LOW_CUT_FREQ/(fs/2),'high');
    ecg = ecg-mean(ecg); % (1) centre
    bpfecg = ecg'-filtfilt(b_lp,a_lp,ecg'); % (2) remove higher freq (zero phase)
    bpfecg = filtfilt(b_bas,a_bas,bpfecg); % (3) remove baseline (zero phase)

    % == remove QRS complex
    residual = mecg_cancellation(qrs,bpfecg,'TS-PCA',5,2,fs);

    % == spectral analysis
    [Pxx,F,~] = pwelch(residual,[],[],[],fs);
    [~,ind] = max(Pxx);
    max_freq = F(ind);
    ratio = sum(Pxx((F>5 & F<9)))/sum(Pxx);

catch
    max_freq = NaN;
    ratio = NaN;
end

end





function residual = mecg_cancellation(peaks,ecg,method,varargin)
nbCycles = 20;
NbPC = 2;
fs = 1000;
switch nargin
    case 3
    case 4
        nbCycles = varargin{1};
    case 5
        nbCycles = varargin{1};
        NbPC = varargin{2};
    case 6
        nbCycles = varargin{1};
        NbPC = varargin{2};
        fs = varargin{3};
    otherwise
        error('mecg_cancellation: wrong number of input arguments \n');
end

% check that we have more peaks than nbCycles
if nbCycles>length(peaks)
    residual = zeros(length(ecg),1);
    disk('MECGcancellation Error: more peaks than number of cycles for average ecg');
    return;
end

% == constants
NB_CYCLES = nbCycles;
NB_MQRS = length(peaks);
ecg_temp = zeros(1,length(ecg));
ecg_buff = zeros(0.7*fs,NB_CYCLES); % ecg stack buffer
Pstart = 0.25*fs-1;
Tstop = 0.45*fs;

try
    % == template ecg (TECG)
    indMQRSpeaks = find(peaks>Pstart);
    for cc=1:NB_CYCLES
        peak_nb = peaks(indMQRSpeaks(cc+1));   % +1 to unsure full cycles
        ecg_buff(:,cc) = ecg(peak_nb-Pstart:peak_nb+Tstop)';
    end
    TECG = median(ecg_buff,2);

    if strcmp(method,'TS-PCA'); [U,~,~] = svds(ecg_buff,NbPC); end;

    % == MECG cancellation
    for qq=1:NB_MQRS
        if peaks(qq)>Pstart && length(ecg)-peaks(qq)>Tstop

            if strcmp(method,'TS')
                % - simple TS -
                ecg_temp(peaks(qq)-Pstart:peaks(qq)+Tstop) = TECG';
            elseif strcmp(method,'TS-SUZANNA')
                % - three scaling factors -
                M  = zeros (0.7*fs,3);
                M(1:0.2*fs,1) = TECG(1:Pstart-0.05*fs+1);
                M(0.2*fs+1:0.3*fs,2) = TECG(Pstart-0.05*fs+2:Pstart+0.05*fs+1);
                M(0.3*fs+1:end,3) = TECG(Pstart+2+0.05*fs:Pstart+1+Tstop);
                a = (M'*M)\M'*ecg(peaks(qq)-Pstart:peaks(qq)+Tstop)';
                ecg_temp(peaks(qq)-Pstart:peaks(qq)+Tstop) = a(1)*M(:,1)'+a(2)*M(:,2)'+a(3)*M(:,3)';

            elseif strcmp(method,'TS-CERUTTI')
                % - only one scaling factor -
                M = TECG;
                a = (M'*M)\M'*ecg(peaks(qq)-Pstart:peaks(qq)+Tstop)';
                ecg_temp(peaks(qq)-Pstart:peaks(qq)+Tstop) = a*M';

            elseif strcmp(method,'TS-LP')
                % - Linear prediction method (Ungureanu et al., 2007) -
                % NOTE: in Ungureanu nbCycles=7
                if qq>NB_CYCLES
                    M = ecg(peaks(qq)-Pstart:peaks(qq)+Tstop)';
                    Lambda = (ecg_buff'*ecg_buff)\ecg_buff'*M;
                    if sum(isnan(Lambda))>0
                        Lambda = ones(length(Lambda),1)/(length(Lambda));
                    end
                    ecg_temp(peaks(qq)-Pstart:peaks(qq)+Tstop) = Lambda'*ecg_buff';
                else
                    M = TECG;
                    ecg_temp(peaks(qq)-Pstart:peaks(qq)+Tstop) = M';
                end

            elseif strcmp(method,'TS-PCA')
                if mod(qq,10)==0
                    % - to allow some adaptation of the PCA basis -
                    % !!NOTE: this adaption step is slowing down the code!!
                    [U,~,~]   = svds(ecg_buff,NbPC);
                end
                % - PCA method -
                X_out  = ecg(peaks(qq)-Pstart:peaks(qq)+Tstop)*(U*U');
                ecg_temp(peaks(qq)-Pstart:peaks(qq)+Tstop) = X_out;
            end

            if qq>NB_CYCLES
               % adapt template conditional to new cycle being very similar to
               % meanECG to avoid catching artifacts. (not used for PCA method).
               Match = CompareCycles(TECG', ecg(peaks(qq)-Pstart:peaks(qq)+Tstop)',0.8);
               if Match
                   ecg_buff = circshift(ecg_buff,[0 -1]);
                   ecg_buff(:,end) = ecg(peaks(qq)-Pstart:peaks(qq)+Tstop)';
                   TECG = median(ecg_buff,2);
               end
            end

        % == managing borders
        elseif peaks(qq)<=Pstart
            % - first cycle if not full cycle -
            n = length(ecg_temp(1:peaks(qq)+Tstop)); % length of first pseudo cycle
            m = length(TECG);                        % length of a pseudo cycle
            ecg_temp(1:peaks(qq)+Tstop+1) = TECG(m-n:end);
        elseif length(ecg)-peaks(qq)<Tstop
            % - last cycle if not full cycle -
            ecg_temp(peaks(qq)-Pstart:end) = TECG(1:length(ecg_temp(peaks(qq)-Pstart:end)));
        end
    end

    % compute residual
    residual = ecg - ecg_temp;

catch ME
    residual = ecg;
end


end

function match = CompareCycles(cycleA,cycleB,thres)
% cross correlation measure to compare if new ecg cycle match with template.
% If not then it is not taken into account for updating the template.
    match = 0;
    bins = size(cycleA,2);
    coeff = sqrt(abs((cycleA-mean(cycleA))*(cycleB-mean(cycleB)))./(bins*std(cycleA)*std(cycleB)));
    if coeff > thres; match = 1; end;
end



