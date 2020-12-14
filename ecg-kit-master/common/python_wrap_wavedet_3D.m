function out = python_wrap_wavedet_3D(input_directory, filename, lead)  
    % Load data.
    filename_matlab = {filename};
    file_tmp=strsplit(filename_matlab{1},'.');
    tmp_input_file = fullfile(input_directory, file_tmp{1});
    [data,header_data] = load_challenge_data(tmp_input_file);
    ecg_lead = (data(lead,:)).';
    %[recording,Total_time,num_leads,Fs,gain,age,sex]=extract_data_from_header(header_data);
    Fs = extract_Fs_from_header(header_data);
    
    % normalize the data for the qrs detector
    ecg_lead = ecg_lead / 100;

    heasig = struct("nsig",1,"freq",500,"nsamp",length(ecg_lead));
    N = length(ecg_lead);
    dt = 1/Fs;
    t = dt*(0:N-1)';
    y = abs(ecg_lead).^2;
    %[~,locs] = findpeaks(y,t,'MinPeakHeight',0.35, 'MinPeakDistance',0.150);
    [~,locs] = findpeaks(y,t,'MinPeakHeight',0.45, 'MinPeakDistance',0.150);
    locs = round(locs * Fs);
    locs = locs.';

    [position,~,~] = wavedet_3D(ecg_lead, locs, heasig, []);
    out = position;
end


function [data,tlines] = load_challenge_data(filename)

        % Opening header file
        disp(filename)
        fid=fopen([filename '.hea']);
        if (fid<=0)
                disp(['error in opening file ' filename]);
        end

        tline = fgetl(fid);
        tlines = cell(0,1);
        while ischar(tline)
            tlines{end+1,1} = tline;
            tline = fgetl(fid);
        end
        fclose(fid);

        f=load([filename '.mat']);
        try
                data = f.val;
        catch ex
                rethrow(ex);
        end

end

function Fs = extract_Fs_from_header(header_data)
	tmp_hea = strsplit(header_data{1},' ');
    Fs = str2num(tmp_hea{3});
end

function [recording,Total_time,num_leads,Fs,gain,age_data,sex_data]=extract_data_from_header(header_data)
	tmp_hea = strsplit(header_data{1},' ');
	recording = tmp_hea{1};
	num_leads = str2num(tmp_hea{2});
    Fs = str2num(tmp_hea{3});
	Total_time = str2num(tmp_hea{4})/Fs;
    gain = zeros(1,num_leads);

	for ii=1:num_leads
	        tmp_hea = strsplit(header_data{ii+1},' ');
            tmp_gain=strsplit(tmp_hea{3},'/');
            gain(ii)=str2num(tmp_gain{1});
    end

    for tline = 1:length(header_data)
        if startsWith(header_data{tline},'#Age')
			tmp = strsplit(header_data{tline},': ');
			age_data = str2num(tmp{2});
        elseif startsWith(header_data{tline},'#Sex')
			tmp = strsplit(header_data{tline},': ');
			if strcmp(tmp{2},'Female')
				sex_data = 1;
			else
				sex_data = 0;
			end
		end
    end
end