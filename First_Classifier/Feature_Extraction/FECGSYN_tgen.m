function [template] = FECGSYN_tgen(ecg_lead,qrs,fs)


if nargin<2; error('ecg_template_build: wrong number of input arguments \n'); end;

ecg = horzcat(ecg_lead{:});
qrs = horzcat(qrs{:});
% == constants
NB_BINS = 250; % number of beans onto which to wrap a cycle
NB_LEADS = size(ecg,1);
NB_SAMPLES = size(ecg,2);
NB_REL = 10; % number relevance. How many cycles minimum to consider that a mode is relevant? - UPDATE ME DEPENDING ON APPLICATION
MIN_NB_CYC = 30; % mininum number of cycles (will decrease THRES until this number of cycles is achieved) - UPDATE ME DEPENDING ON APPLICATION
THRES = 0.9; % threshold at which to decide whether the cycle match or not - UPDATE ME DEPENDING ON APPLICATION
MIN_TLEN = 0.35;     % minimum template length in ms - UPDATE ME DEPENDING ON APPLICATION
PACE = 0.1;
MIN_THRES = 0.6;
cycle = zeros(NB_LEADS,NB_BINS);
startCycle = 2;
NbModes = 1; % initialisation variable
relevantModeInd = []; % - UPDATE ME DEPENDING ON APPLICATION
relevantMode.NbCycles = 0;

% == linear phase wrapping (shift in -pi/6)
qrs = qrs(qrs > 0);
phase = FECGSYN_kf_phasecalc(qrs,NB_SAMPLES);
PhaseChangePoints = find(phase(2:end)<0&phase(1:end-1)>0);
NB_CYCLES = length(PhaseChangePoints);

% ==== core function ====
% == creating the different modes
cycleIndex = (PhaseChangePoints(startCycle)+1:PhaseChangePoints(startCycle+1));
for j=1:NB_LEADS
    cycle(j,:) = interp1(phase(cycleIndex),ecg(j,cycleIndex),linspace(-pi,pi,NB_BINS),'spline');
end
Mode{NbModes}.cycles = zeros(NB_LEADS,NB_BINS,1);       % cycles included
Mode{NbModes}.cycles(:,:,1) = cycle;
Mode{NbModes}.cycleMean = cycle;                        % average cycle
Mode{NbModes}.cycleStd = zeros(3,NB_BINS);              % standard deviation from cycles
Mode{NbModes}.NbCycles = 1;                             % number of cycles present
Mode{NbModes}.cycleLen = [];                            % length of cycles (in samples)

while relevantMode.NbCycles<MIN_NB_CYC && THRES>MIN_THRES
    % the THRES is lowered until a mode with mode than MIN_NB_CYC cycles is
    % detected or the THREShold is too low (which means no mode can be identified)
    for i=startCycle+1:NB_CYCLES-2
        cycleIndex = (PhaseChangePoints(i)+1:PhaseChangePoints(i+1));
        for j=1:NB_LEADS
            cycle(j,:) = interp1(phase(cycleIndex),ecg(j,cycleIndex),linspace(-pi,pi,NB_BINS),'spline');
        end
        match = 0;
        indMode = 1;
        while (~match&indMode<=NbModes)
%             [r,~] = corrcoef(cycle,Mode{indMode}.cycleMean);
%             coeff = abs(r(1,2));
            % 50% computation time
            coeff = corrcoef(cycle,Mode{indMode}.cycleMean);
            match = coeff>THRES;
            indMode = indMode+1;
        end
        if ~match  % if the new cycle does not match with the average cycle of any mode
            % then create a new mode
            NbModes=NbModes+1;
            Mode{NbModes}.cycles = zeros(NB_LEADS,NB_BINS,1);
            Mode{NbModes}.cycles(:,:,1) = cycle;
            Mode{NbModes}.cycleMean = cycle;
            Mode{NbModes}.cycleStd = zeros(3,NB_BINS);
            Mode{NbModes}.NbCycles = 1;
            Mode{NbModes}.cycleLen = length(cycleIndex);
        else % it it correlates then integrate it to the corresponding mode
            Mode{indMode-1}.NbCycles = Mode{indMode-1}.NbCycles+1;
            temp = Mode{indMode-1}.cycles ;
            Mode{indMode-1}.cycles = zeros(NB_LEADS,NB_BINS,Mode{indMode-1}.NbCycles);
            Mode{indMode-1}.cycles(:,:,1:end-1)= temp;
            Mode{indMode-1}.cycles(:,:,end)= cycle;
            Mode{indMode-1}.cycleMean = mean(Mode{indMode-1}.cycles,3);
            Mode{indMode-1}.cycleStd = std(Mode{indMode-1}.cycles,0,3);
            Mode{indMode-1}.cycleLen = [Mode{indMode-1}.cycleLen length(cycleIndex)];
        end

    end

    % == detecting what mode is relevant
    %   relevantMode:   structure containing cycle, cycleMean and cycleStd
    %                   representing how many cycles have been selected to build the stack, the
    %                   mean ecg cycle that is built upon these selected cycles and the
    %                   standard deviation for each point of the template cycle as an indicator
    %                   of the precision of the estimation. *Only the dominant mode is outputted
    %                   for this application.*
    for i=1:length(Mode)
        % minimum amount of cycles and length
        if Mode{i}.NbCycles>NB_REL && mean(Mode{i}.cycleLen) >= MIN_TLEN*fs
            relevantModeInd = [relevantModeInd i];
        end
    end
    relevantMode = Mode(relevantModeInd);

    if isempty(relevantMode)
        % if we did not catch anything then output rubbish
        relevantMode.cycleMean = ones(NB_BINS,1);
        relevantMode.cycleStd = ones(NB_BINS,1);
        relevantMode.NbCycles  = 0;
        relevantMode.cycleLen = 0;
        status = 0;
        template.avg = NaN;
        template.stdev = NaN;
        qrsloc = NaN;
    else
        relevantModeStruc = [relevantMode{:}];
        [~,pos] = max([relevantModeStruc.NbCycles]); % look for dominant mode
        relevantMode = relevantMode{pos}; % only return the dominant mode for this application
        status = 1;

        % == Converting template from bins back to samples
        desl = round(NB_BINS/6);
        template.avg = circshift(relevantMode.cycleMean',-desl);
        template.stdev = circshift(relevantMode.cycleStd',-desl);
        template.avg = resample(template.avg,round(mean(relevantMode.cycleLen)),NB_BINS);
        template.stdev = resample(template.avg,round(mean(relevantMode.cycleLen)),NB_BINS)';
        qrsloc = round(round((NB_BINS/2 - desl)*(length(template.avg)/NB_BINS)));
        [~,delay]=max(abs(template.avg));
        if abs(qrsloc-delay)<3, qrsloc = delay; end  % allow some alignment
    end
    THRES = THRES-PACE;
end



end


function phase = FECGSYN_kf_phasecalc(peaks,NbSamples)

phase = zeros(1,NbSamples);
m = diff(peaks);

% first interval uses second interval as reference
L = peaks(1);       %length of first interval
if isempty(m)       % only ONE peak was detected
    phase(1:NbSamples) = linspace(-2*pi,2*pi,NbSamples);
else
    phase(1:L) = linspace(2*pi-L*2*pi/m(1),2*pi,L);
    % beats in the middle
    for i = 1:length(peaks)-1;      % generate phases between 0 and 2pi for almos all peaks
        phase(peaks(i):peaks(i+1)) = linspace(0,2*pi,m(i)+1);
    end                             % 2pi is overlapped by 0 on every loop
    L = length(phase)-peaks(end);   %length of last interval
    phase(peaks(end):end) = linspace(0,L*2*pi/m(end),L+1);
end
phase = mod(phase,2*pi);
phase(phase>pi) = phase(phase>pi)- 2*pi;
end