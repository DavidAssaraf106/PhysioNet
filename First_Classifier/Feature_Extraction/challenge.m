function [output] = challenge(ecg, lead, fs, qrs_peak, position)


%addpath(genpath('hrv_Aviv_toolbox/'));
%addpath(genpath('ecg-kit-0.1.0/'));

ecg = horzcat(ecg{:});
qrs_peak = horzcat(qrs_peak{:});
classifyResult = 'N'; % default output normal rhythm
USE_COMPUTED_QRS = 0;

if ~exist('opt')


opt.THRES_SQI = 0.9229;
opt.WIND_SQI = 0.0796;
opt.REG_WIN = 1;
opt.SIZE_WIND = 3;
opt.LG_MED = 1;
opt.THERS_TAC = 101;
opt.THERS_BRA = 49;
opt.THRES_AF = -1.2318;
opt.MIN_AF_RR = 5;
opt.WIND_TOL = 27.987;
opt.MIN_NB_FILT = 10;
opt.JQRS_THRESH = 0.6;
opt.JQRS_REFRAC = 0.15;
opt.JQRS_INTWIN_SZ = 7;
opt.MIN_SUB_SEG_TIME = 7;
opt.False_Outlier_Thres = -0.1027;
opt.WIND_NB_SAMP = 12;
opt.sqi_outliers = 0.8874;
opt.bimod_alt = 0.91;
opt.bimod_delta_perc = 0.3916;
opt.bimod_delta = 18.9001;

end

jqrs = qrs_peak; epltd = []; anns_gqrs = []; anns_wqrs = [];
NB_PT = length(ecg);

%%
%% BEGIN NEW
% == ECG segmentation
% patch waiting for the constants
% ecg_select = ecg(round(start_seg*fs)+1:round(end_seg*fs));
% qrs_select = jqrs(jqrs>start_seg*fs & jqrs<end_seg*fs)-ceil(start_seg*fs);
heasig = struct("nsig",1,"freq",500,"nsamp",length(ecg));
ecg_select = ecg;
qrs_select = double(horzcat(jqrs));
% ecg_select_res = resample(ecg_select, double(300), double(fs)); % à corriger avec Joachim
ecg_select_res = ecg;
qrs_select_res = round(qrs_select*double(300)/double(fs));
% [position,~,~] = wavedet_3D(ecg_select_res, qrs_select_res, heasig, []);
% [position,~,~] = wavedet_3D(ecg_select, [], heasig, []);
% == baseline wander removal
[b_bas,a_bas] = butter(2,0.7/(300/2),'high');
bp_ecg_select_res = filtfilt(b_bas,a_bas,ecg_select_res);
bp_ecg_select_res = filtfilt(b_bas,a_bas,bp_ecg_select_res);

% ====> NEW Looking if we have a continuous HR above or below tachy/brady
% threshold and if yes take it otherwise take the median HR over the whole
% signal

fs=double(fs);

% == f-wave detection for P-wave
[ratio,max_freq] = f_wave_detection(ecg_select,qrs_select,fs);

% == QRS length
[~,~,~,~,R_amp_m,R_amp_std] = compute_qrs_size(ecg_select, round(qrs_select), fs, 0);

% ====> UPDATED Fix to Q and S fiducials
ind_Q_vec = round(position.QRSon*(fs/300))';
ind_S_vec = round(position.QRSoff*(fs/300))';
QS = nanmedian(ind_S_vec - ind_Q_vec)/fs;
QS_std = nanstd(ind_S_vec - ind_Q_vec)/fs;


% == getting all the morphological intervals
medRR = median(diff(qrs_select)/fs);
if ~isempty(position)
    medQT =  nanmedian((position.Toff'/300-ind_Q_vec/fs)); % important use nanmedian
    [medQT_b, medQT_fre, medQT_fra, medQT_hod] = corrected_qt(medQT, medRR);
    medP = nanmedian((position.Poff - position.Pon)/300); % median P-wave length
    stdP = nanstd((position.Poff - position.Pon)/300); % std of P-wave length
    medT = nanmedian((position.Toff - position.Ton)/300); % median T-wave length
    stdT = nanstd((position.Toff - position.Ton)/300); % std of T-wave length
    medPR = nanmedian(ind_Q_vec'/fs-position.Pon/300); % median PR interval length
    stdPR = nanstd(ind_Q_vec'/fs - position.Pon/300); % std of PR interval length
    medPRseg = nanmedian(ind_Q_vec'/fs-position.Poff/300); % median PR segment length
    indTamp = find(~isnan(position.T) & ~isnan(position.Toff)); % check beats for which we have both T amplitude and Toff fiducials
    medTamp = nanmedian(bp_ecg_select_res(position.T(indTamp)) - bp_ecg_select_res(position.Toff(indTamp))); % median T-wave amplitude (taken as max - Tend)
    stdTamp = nanstd(bp_ecg_select_res(position.T(indTamp)) - bp_ecg_select_res(position.Toff(indTamp))); % std T-wave amplitude
    medTtype = median(position.Ttipo); % median T-wave type (1:5 corresponding respectively to normal, inverted, upwards only, downwards only, biphasc pos-neg, biphasc neg-pos)

    nbpwaves = length(~isnan(position.Poff - position.Pon))*medRR/length(ecg); % NEW UPDATED number of P-wave detected by unit of time normalised by the heart rate. Idea is that less P-wave will be detected in AF cases
    indPamp = find(~isnan(position.P) & ~isnan(position.Poff)); % beats for which we have a P amplitude and Poff fiducials
    medPamp = nanmedian(bp_ecg_select_res(position.P(indPamp)) - bp_ecg_select_res(position.Poff(indPamp))); % NEW median P-wave amplitude computed from peak P-wave to Poff
    stdPamp = nanstd(bp_ecg_select_res(position.P(indPamp)) - bp_ecg_select_res(position.Poff(indPamp))); % NEW std of P-wave amplitude
    
    indSTamp = find(~isnan(position.QRSoff) & ~isnan(position.Ton));
    medST = nanmedian((position.Ton(indSTamp)-position.QRSoff(indSTamp))/300); % NEW: ST segment length. S point starting at QRSoff and finishing at Ton
    
    % == NEW: ST segment elevation/depletion
    indPRbas = find(~isnan(position.Poff) & ~isnan(position.QRSon) & ~isnan(position.QRSoff) & ~isnan(position.Ton)); % only select cycles for which we get the Poff AND QRSon AND QRSoff AND Ton
    nbPRbas = length(indPRbas);
    medJTamp = zeros(1,nbPRbas);
    medJT2amp = zeros(1,nbPRbas);
    medPRamp = zeros(1,nbPRbas);
    for kk=1:nbPRbas
        medJTamp(kk) = nanmedian(bp_ecg_select_res(position.QRSoff(indPRbas(kk)):position.Ton(indPRbas(kk)))); % median amplitude QRSoff->Ton ('J point') = ~ST
        medJT2amp(kk) = nanmedian(bp_ecg_select_res(position.QRSoff(indPRbas(kk))+round(0.060*300):position.Ton(indPRbas(kk)))); % median amplitude QRSoff+60 ms ('J point')->Ton = ~ST
        medPRamp(kk) = nanmedian(bp_ecg_select_res(position.Poff(indPRbas(kk)):position.QRSon(indPRbas(kk)))); % Poff to QRSon - median PR interval amplitude
    end
    medSTvar1 = median(medJTamp) - median(medPRamp); % J point
    medSTvar2 = nanmedian(medJT2amp) - median(medPRamp); % J point + 60 ms, nanmedian for JT2amp because of the + 60 ms which might end up being after the Ton sometime
    stdSTvar1 = nanstd(medJTamp - medPRamp);
    stdSTvar2 = nanstd(medJT2amp - medPRamp);
    
        % == NEW: checking how many time we get all the fiducials
    indALL = find(~isnan(position.Poff) & ~isnan(position.Pon) &... 
        ~isnan(position.QRSon) & ~isnan(position.QRSoff) & ...
        ~isnan(position.Toff) & ~isnan(position.Ton));
    nbAllwaves = length(indALL)/length(position.QRSon); % proportion of cycles for which we manage to make a full segmentation
else
    disp('Not able to extract morphological features');
    medQT = NaN;
    medQT_b = NaN;
    medQT_fre = NaN;
    medQT_fra = NaN;
    medQT_hod = NaN;
    medP = NaN;
    medPR = NaN;
    stdPR = NaN;
    medT = NaN;
    medTamp = NaN;
    stdTamp = NaN;
    medTtype = NaN;
    stdP = NaN;
    stdT = NaN;
    nbpwaves = NaN;
    medPamp = NaN;
    stdPamp = NaN;
    medSTvar1 = NaN;
    medSTvar2 = NaN;
    medST = NaN;
    stdSTvar1 = NaN;
    stdSTvar2 = NaN;
    nbALLwaves = NaN;
    medPRseg = NaN;
end

s = int2str(lead);

output = struct(strcat('QS_',s),abs(QS), strcat('R_amp_m_',s),abs(R_amp_m) , strcat('R_amp_std_',s),abs(R_amp_std),...
    strcat('ratio_',s),abs(ratio) ,...
    strcat('medQT_',s),abs(medQT) , strcat('medQT_b_',s), abs(medQT_b) , strcat('medQT_fre_',s),abs(medQT_fre) , strcat('medQT_hod_',s),abs(medQT_hod) ,...
    strcat('medPR_',s),abs(medPR) , strcat('stdPR_',s),abs(stdPR),...
    strcat('medTtype_',s), abs(medTtype) , strcat('stdP_',s),abs(stdP), strcat('stdT_',s),abs(stdT) ,...
    strcat('stdPamp_',s),abs(stdPamp),  abs(medSTvar1) , strcat('medSTvar2_',s),abs(medSTvar2),...
    strcat('medPRseg_',s), abs(medPRseg));


end





function [ratio, max_freq] = f_wave_detection(ecg,qrs,fs)
% this function compute the dominant frequency mode in the badn 4-45 Hz
% after subtraction of the ECG cycle.
%
% input
%   ecg: ecg time series
%   qrs: R peak location
%   fs: sampling frequency
%
% ouput
%   ratio: ratio of power in the band 5-9 Hz versus total power
%   max_freq: dominant frequency
%
% Copyright (C) 2017 Joachim A. Behar
% Bio-electric and Bio-energetic system Group, Technion-IIT, Israel
% joachim.a.behar@gmail.com
%
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 2 of the License, or (at your
% option) any later version.
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
% Public License for more details.

try
    % == prefiltering
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


function [QT_b, QT_fre, QT_fra, QT_hod] = corrected_qt(medQT, medRR)
% Copyright (C) 2017 Joachim A. Behar
% Bio-electric and Bio-energetic system Group, Technion-IIT, Israel
% joachim.a.behar@gmail.com
%
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 2 of the License, or (at your
% option) any later version.
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
% Public License for more details.

QT_b = medQT/sqrt(medRR);
QT_fre = medQT/medRR.^3;
QT_fra = medQT+0.154*(1-medRR);
%QT_hod = medQT+1.75*(60/medRR-60);
QT_hod = medQT+0.00175*(60/medRR-60);

end


function [ind_Q_vec,ind_S_vec,QS,QS_std,R_amp_m,R_amp_std] = compute_qrs_size(ecg,jqrs,fs,debug)
% This function computes the QS interval and R peak amplitude. It only
% takes into account these interval/hight in the interval of good quality.
%
% inputs
%   ecg: raw ecg
%   jqrs: jqrs R-peak annotations
%   sqi: second by second signal quality
%   fs: sampling frequency
%
% outputs
%   ind_Q_vec: index of Q fiducials
%   ind_S_vec: index of S fiducials
%   QS: median QS interval
%   QS_std: standard deviation on QS intervals
%   R_amp: median R peak amplitude
%   R_amp:_std: std on R peak amplitude
%
% Copyright (C) 2017 Joachim A. Behar
% Bio-electric and Bio-energetic system Group, Technion-IIT, Israel
% joachim.a.behar@gmail.com
%
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 2 of the License, or (at your
% option) any later version.
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
% Public License for more details.

delta_s = 0.150*fs;
NB_QRS = length(jqrs);

% == prefiltering
LOW_CUT_FREQ = 2;
HIGH_CUT_FREQ = 45;
[b_lp,a_lp] = butter(5,double(HIGH_CUT_FREQ)/(double(fs)/2),'high');
[b_bas,a_bas] = butter(2,double(LOW_CUT_FREQ)/(double(fs)/2),'high');
ecg = ecg-mean(ecg);
bpfecg = ecg'-filtfilt(b_lp,a_lp,ecg');
bpfecg = filtfilt(b_bas,a_bas,bpfecg);
if sum(sign(bpfecg(jqrs(jqrs > 0))))<0
    bpfecg = -bpfecg;
end

ind_Q_vec = zeros(NB_QRS,1);
ind_S_vec = zeros(NB_QRS,1);
R_amp = zeros(NB_QRS,1);

% = loop over each R-peak
for kk=1:NB_QRS

    if jqrs(kk)-delta_s>0 && jqrs(kk)+delta_s<length(ecg)

        % check border are okay and that the sqi is 1
        %[~,ind_Q] = min(bpfecg(jqrs(kk)-delta_s:jqrs(kk)));
        %[~,ind_S] = min(bpfecg(jqrs(kk):jqrs(kk)+delta_s));
        %ind_Q_vec(kk) = jqrs(kk)-delta_s+ind_Q-1;
        %ind_S_vec(kk) = jqrs(kk)+ind_S-1;

        [~,ind_Q] = findpeaks(-bpfecg(jqrs(kk)-delta_s:jqrs(kk)));
        [~,ind_S] = findpeaks(-bpfecg(jqrs(kk):jqrs(kk)+delta_s));
        ind_Q(delta_s-ind_Q<fs*0.02) = [];
        ind_S(ind_S<fs*0.02) = [];

        R_amp(kk) = ecg(jqrs(kk));
        if isempty(ind_Q) || isempty(ind_S)
            ind_Q_vec(kk) = NaN;
            ind_S_vec(kk) = NaN;
        else
            ind_Q_vec(kk) = jqrs(kk)-delta_s+ind_Q(end)-1;
            ind_S_vec(kk) = jqrs(kk)+ind_S(1)-1;
        end

    else
        ind_Q_vec(kk) = NaN;
        ind_S_vec(kk) = NaN;
    end
end

QS = nanmedian(ind_S_vec-ind_Q_vec)/fs;
QS_std = nanstd(ind_S_vec-ind_Q_vec)/fs;
R_amp_m = nanmedian(R_amp);
R_amp_std = nanstd(R_amp);

if debug
    plot(ecg);
    hold on; plot(bpfecg,'--r');
    hold on; plot(jqrs,ecg(jqrs),'+r')
    hold on; plot(ind_Q_vec(~isnan(ind_Q_vec)),ecg(ind_Q_vec(~isnan(ind_Q_vec))),'+g');
    hold on; plot(ind_S_vec(~isnan(ind_S_vec)),ecg(ind_S_vec(~isnan(ind_S_vec))),'+k');
    legend('ecg','filtered ecg','Q','S');
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



