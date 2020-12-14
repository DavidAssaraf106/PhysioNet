%% Gets the format of an ECG recording filename
% This function tries to get the "fingerprints" of several ECG formats in
% order to do format "auto-detection"
% 
% Example
% 
%   recording_format = ECGformat(recording_filename)
% 
% See also read_ECG, ECGwrapper
% 
% Author: Mariano Llamedo Soria
% <matlab:web('mailto:llamedom@electron.frba.utn.edu.ar','-browser') (email)> 
% Version: 0.1 beta
% Birthdate: 5/01/2014
% Last update: 19/11/2014
% Copyright 2008-2015
% 
function recording_format = ECGformat(recording_filename)
    
    recording_format = [];

    cKnownFormats = {'MIT' 'ISHNE', 'AHA', 'HES', 'MAT', 'Mortara', 'HL7a'};
    matformat_definitions();
    lcKnownFormats = length(cKnownFormats);

    if( nargin < 0 || ~ischar(recording_filename) )
        error('ECGformat:UnknownDataFormat', 'recording_filename must be a char array.\n')
    end
    
    [rec_filepath, rec_filename, rec_fileExt] = fileparts(recording_filename);

    aux_idx = 1:lcKnownFormats;

    % Mortara format is a folder with predefined files inside named
    % HourXXRawData.bin
    if( isdir(recording_filename) )
        rec_path = recording_filename;
        hours_files = dir([rec_path filesep 'Hour*.bin' ]);
        if( isempty(hours_files) )
            % empty folder, not Mortara
            return
        else
            recording_format = 'Mortara';
            return
        end
    else
        [~, rec_name] = fileparts(recording_filename);
        if( length(rec_name) > 7 && strcmpi( rec_name(1:4), 'Hour') && strcmpi( rec_name(end-6:end), 'RawData') )
            recording_format = 'Mortara';
            return
        end
    end
    
    
    % first guess based on typical file extensions
    if( strcmpi(rec_fileExt, '.dat' ) )
        % MIT first
        fg_idx = find(strcmpi(cKnownFormats, 'MIT'));
        aux_idx = [fg_idx aux_idx(aux_idx~=fg_idx)];
    elseif( strcmpi(rec_fileExt, '.ecg' ) )
        % ISHNE and AHA 
        fg_idx = find(strcmpi(cKnownFormats, 'ISHNE'));
        aux_idx = [fg_idx aux_idx(aux_idx~=fg_idx)];
        fg_idx = find(strcmpi(cKnownFormats, 'AHA'));
        aux_idx = [fg_idx aux_idx(aux_idx~=fg_idx)];
    elseif( strcmpi(rec_fileExt, '.mat' ) )
        % Matlab 
        fg_idx = find(strcmpi(cKnownFormats, 'MAT'));
        aux_idx = [fg_idx aux_idx(aux_idx~=fg_idx)];
    elseif( strcmpi(rec_fileExt, '.xml' ) )
        % Matlab 
        fg_idx = find(strcmpi(cKnownFormats, 'HL7a'));
        aux_idx = [fg_idx aux_idx(aux_idx~=fg_idx)];
    elseif( strcmpi(rec_fileExt, '.hes' ) )
        % HES 
        fg_idx = find(strcmpi(cKnownFormats, 'HES'));
        aux_idx = [fg_idx aux_idx(aux_idx~=fg_idx)];
    end
    
    for ii = aux_idx

        if( strcmpi(cKnownFormats{ii}, 'MIT') )
            
            header_fn = [rec_filepath filesep rec_filename '.hea'];
            if( exist( header_fn, 'file' ) )
                aux_header = readheader(header_fn);
                if( ~isempty(aux_header) && strcmpi(aux_header.recname, rec_filename) )
                    recording_format = cKnownFormats{ii};
                    return
                end
            end

        elseif( strcmpi(cKnownFormats{ii}, 'ISHNE') )
            
            if(isISHNEformat(recording_filename))
                recording_format = cKnownFormats{ii};
                return
            end

        elseif( strcmpi(cKnownFormats{ii}, 'HL7a') )
            
            if(isHL7aformat(recording_filename))
                recording_format = cKnownFormats{ii};
                return
            end

        elseif( strcmpi(cKnownFormats{ii}, 'AHA') )
            
            if(isAHAformat(recording_filename))
                recording_format = cKnownFormats{ii};
                return
            end

        elseif( strcmpi(cKnownFormats{ii}, 'HES') )

            if(isHESformat(recording_filename))
                recording_format = cKnownFormats{ii};
                return
            end
            
        elseif( strcmpi(cKnownFormats{ii}, 'MAT') )
            
            try
                
                aux_load = load(recording_filename);

                fnames = fieldnames(aux_load);
                signal_name = intersect( fnames, cMatSignalNames);
                header_name = intersect( fnames, cMatSignalHeaderNames);
                
                if( ~isempty(header_name) && ~isempty(signal_name) )
                    recording_format = cKnownFormats{ii};
                    return
                end

            catch ME
                
            end

        end

    end
    
    