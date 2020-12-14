classdef ECGtask_QRS_detection < ECGtask

% ECGtask for ECGwrapper (for Matlab)
% ---------------------------------
% 
% Description:
% 
% Abstract class for defining ECGtask interface
% 
% Adding user-defined QRS detectors:
% A QRS detector that has the following interface can be added to the task:
% 
%     [positions_single_lead, position_multilead] = your_QRS_detector( ECG_matrix, ECG_header, progress_handle, payload_in);

% where the arguments are:
%    + ECG_matrix, is a matrix size [ECG_header.nsamp ECG_header.nsig]
%    + ECG_header, is a struct with info about the ECG signal, such as:
%         .freq, the sampling frequency
%         .desc, description about the signals.
%    + progress_handle, is a handle to a waitbar object, that can be used
%          to track the progress within your function. See the
%          documentation about this class in this kit.
%    + payload_in, is a user data variable allowed to be sent each call to
%          your function. It is sent, via the payload property of this
%          class, for example:
% 
%         this_ECG_wrappers.ECGtaskHandle.payload = your_variable;
%         this_ECG_wrappers.ECGtaskHandle.payload = {your_var1 your_var2};
%         this_ECG_wrappers.ECGtaskHandle.payload = load(cached_filenames);
% 
% the output of your function must be:
%    + positions_single_lead, a cell array size ECG_header.nsig with the
%          QRS sample locations found in each lead.
%    + position_multilead, a numeric vector with the QRS locations
%          calculated using multilead rules.
% 
% Author: Mariano Llamedo Soria (llamedom at frba.utn.edu.ar)
% Version: 0.1 beta
% Birthdate  : 18/2/2013
% Last update: 18/2/2013
% Copyright 2008-2017
      
    properties(GetAccess = public, Constant)
        name = 'QRS_detection';
        target_units = 'ADCu';
        doPayload = true;
    end

    properties( GetAccess = public, SetAccess = private)
        % if user = memory;
        % memory_constant is the fraction respect to user.MaxPossibleArrayBytes
        % which determines the maximum input data size.
        memory_constant = 0.3;
        
        started = false;
        
    end
    
    properties( Access = private, Constant)
        
        cQRSdetectors = {'all-detectors' 'wavedet' 'pantom' 'aristotle' 'gqrs' 'sqrs' 'wqrs' 'ecgpuwave' 'epltdqrs1' 'epltdqrs2' };
        WFDBMaxSampRate = 260; % Hz
        
    end
    
    properties( Access = private )
        
        detectors2do
        bWFDBdetectors
        WFDBtarget_samp_rate = 250; %Hz
        WFDBResampled = false;
        tmp_path_local
        WFDB_bin_path
        WFDB_cmd_prefix_str 
        lead_names = [];
        wavedet_config
        
    end
    
    properties
        progress_handle
        tmp_path
        lead_idx = [];
        detectors = 'all-detectors';
        only_ECG_leads = false;
        gqrs_config_filename = [];
        payload
        CalculatePerformance = false;
        detection_threshold = 1;
        bRecalculateNewDetections = false;
        bRecalcQualityAndPerformance = false;
        
    end
    
    methods
           
        function obj = ECGtask_QRS_detection(obj)

            obj.wavedet_config.setup.wavedet.QRS_detection_only = true;

        end
        
        function Start(obj, ECG_header, ECG_annotations)

            if( isempty(obj.lead_idx) )
                
                if( obj.only_ECG_leads )
                    obj.lead_idx = get_ECG_idx_from_header(ECG_header);
                else
    %                 'all-leads'
                    obj.lead_idx = 1:ECG_header.nsig;
                end
                
            else
                obj.lead_idx( obj.lead_idx > ECG_header.nsig ) = ECG_header.nsig;
                obj.lead_idx = unique(obj.lead_idx);
            end
            
            if( isempty(obj.lead_idx) )
                return
            end
            
            if( any(strcmpi('all-detectors', obj.detectors)) )
                obj.detectors2do = obj.cQRSdetectors(2:end);
            else
                if( ischar(obj.detectors) )
                    obj.detectors2do = cellstr(obj.detectors);
                else
                    obj.detectors2do = obj.detectors;
                end
            end

            % lead names desambiguation
            str_aux = regexprep(cellstr(ECG_header.desc), '\W*(\w+)\W*', '$1');
            obj.lead_names = regexprep(str_aux, '\W', '_');

            [str_aux2, ~ , aux_idx] = unique( obj.lead_names );
            aux_val = length(str_aux2);
            
            if( aux_val ~= ECG_header.nsig )
                for ii = 1:aux_val
                    bAux = aux_idx==ii;
                    aux_matches = sum(bAux);
                    if( sum(bAux) > 1 )
                        obj.lead_names(bAux) = strcat( obj.lead_names(bAux), repmat({'v'}, aux_matches,1), cellstr(num2str((1:aux_matches)')) );
                    end
                end
            end
            
            obj.lead_names = regexprep(obj.lead_names, '\W*(\w+)\W*', '$1');
            obj.lead_names = regexprep(obj.lead_names, '\W', '_');
            
            
            obj.bWFDBdetectors = ~isempty(intersect({'aristotle' 'gqrs' 'sqrs' 'wqrs' 'ecgpuwave' 'epltdqrs1' 'epltdqrs2'}, obj.detectors2do));
            
            % local path required to avoid network bottlenecks in distributed filesystems 
            if( isunix() && exist('/scratch/', 'dir') )
                str_username = getenv('USER');
                obj.tmp_path_local = ['/scratch/' str_username filesep];
                if( ~exist(obj.tmp_path_local, 'dir') )
                    if(~mkdir(obj.tmp_path_local))
                        obj.tmp_path_local = '/scratch/';
                    end
                end
                obj.tmp_path = tempdir;
            else
                if( isempty(obj.tmp_path) )
                    obj.tmp_path = tempdir;
                end
                obj.tmp_path_local = obj.tmp_path;
            end
            
            obj.WFDB_bin_path = init_WFDB_library(obj.tmp_path_local);
            
            obj.WFDB_cmd_prefix_str = WFDB_command_prefix(obj.tmp_path_local);
            
            obj.started = true;
        end
        
        function payload_out = Process(obj, ECG, ECG_start_offset, ECG_sample_start_end_idx, ECG_header, ECG_annotations, ECG_annotations_start_end_idx  )
            
            payload_out = [];

            if( ~obj.started )
                obj.Start(ECG_header);
                if( ~obj.started )
                    cprintf('*[1,0.5,0]', 'Task %s unable to be started for %s.\n', obj.name, ECG_header.recname);
                    return
                end
            end
            
            % Signal conversion to MIT format to be used by WFDB detectors.
            if(obj.bWFDBdetectors)
                % MIT conversion is needed for WFDB detectors.
                % Resampled to obj.WFDBtarget_samp_rate Hz for performance
                
                ECG_header_WFDB = ECG_header;
                ECG_header_WFDB.recname = regexprep(ECG_header_WFDB.recname, '\W*(\w+)\W*', '$1');
                ECG_header_WFDB.recname = regexprep(ECG_header_WFDB.recname, '\W', '_');
                
                MIT_filename = [ECG_header_WFDB.recname '_' num2str(ECG_start_offset) '_' num2str(ECG_header_WFDB.nsamp+ECG_start_offset-1) ];
                ECG_header_WFDB.recname = MIT_filename;
                
                if(ECG_header_WFDB.freq > obj.WFDBMaxSampRate )
                    [p, q] = rat(obj.WFDBtarget_samp_rate/ECG_header_WFDB.freq);
                    obj.WFDBResampled = true;
                    ECG_header_WFDB.freq = round(ECG_header_WFDB.freq * p / q);
                else
                    obj.WFDBResampled = false;
                end
                
                MIT_filename = [obj.tmp_path_local MIT_filename '.dat'];
                fidECG = fopen(MIT_filename, 'w');
                try
                    if( obj.WFDBResampled )
                        fwrite(fidECG, int16(round(resample(double(ECG), p, q)))', 'int16', 0 );
                    else
                        fwrite(fidECG, ECG', 'int16', 0 );
                    end
                    
                    fclose(fidECG);
                catch MEE
                    fclose(fidECG);
                    rethrow(MEE);
                end

                writeheader(obj.tmp_path_local, ECG_header_WFDB);   

                if( isempty(obj.gqrs_config_filename) )
                    aux_str = obj.WFDB_bin_path;
                    if( aux_str(end) == filesep )
                        obj.gqrs_config_filename = [aux_str 'gqrs.conf' ];
                    else
                        obj.gqrs_config_filename = [aux_str filesep 'gqrs.conf' ];
                    end
                end
                
            end

            detectors_calculated = [];

            if( obj.bRecalcQualityAndPerformance )
            
                % payload property is used in this task to input an external QRS
                % detector, or manually corrected detections.
                if( isstruct(obj.payload) )

                    aux_struct = obj.payload;
                    
                    if( isfield(aux_struct, 'series_quality') && isfield(aux_struct.series_quality, 'sampfreq' ) )
                        ann_sampfreq = aux_struct.series_quality.sampfreq;
                    else
                        % assumed in the same sampling rate
                        ann_sampfreq = ECG_header.freq;
                    end
                                        
                    payload_out.series_quality.sampfreq = ECG_header.freq;
                
                    % ratio to convert annotations @ ann_sampfreq to ECG_header.freq
                    aux_sampfreq_ratio = ECG_header.freq / ann_sampfreq;

                    if( isfield(aux_struct, 'series_quality') )
                        
                        detectors_calculated = aux_struct.series_quality.AnnNames(:,1);
                        
                        for ii = 1:size(aux_struct.series_quality.AnnNames,1)
                            aux_val = unique( round(aux_struct.(aux_struct.series_quality.AnnNames{ii,1}).(aux_struct.series_quality.AnnNames{ii,2}) * aux_sampfreq_ratio ) ) - ECG_start_offset + 1;
                            aux_val = aux_val( aux_val >= ECG_sample_start_end_idx(1) & aux_val <= ECG_sample_start_end_idx(2) );
                            aux_val = aux_val + ECG_start_offset - 1;
                            payload_out.(aux_struct.series_quality.AnnNames{ii,1}).(aux_struct.series_quality.AnnNames{ii,2}) = aux_val;
                        end

                    else
                        fnames = fieldnames(obj.payload);
                        for fname = rowvec(fnames)
                            if( isfield(aux_struct.(fname{1}), 'time') )
                                aux_val = unique( round( aux_struct.(fname{1}).time * aux_sampfreq_ratio ) ) - ECG_start_offset + 1;
                                aux_val = aux_val( aux_val >= ECG_sample_start_end_idx(1) & aux_val <= ECG_sample_start_end_idx(2) );
                                aux_val = aux_val + ECG_start_offset - 1;
                                payload_out.(fname{1}).time = aux_val;
                                detectors_calculated = [detectors_calculated; fname];
                            end
                        end
                    end

                    detectors_calculated = unique(cellfun( @(a)(a{1}), regexp(detectors_calculated, '(.+)_.+', 'tokens')));
                    
                else
                    cprintf('*[1,0.5,0]', 'Unable to parse payload struct for %s.\n', ECG_header.recname);
                    return                    
                end                

            end
            
            if( obj.bRecalculateNewDetections )
                % If recalculate only new detections, we exclude
                % detectors already processed.
                obj.detectors2do = setdiff(obj.detectors2do, detectors_calculated);
            end

            % Actual QRS detection
            cant_QRSdetectors = length(obj.detectors2do);

            for ii = 1:cant_QRSdetectors

                this_detector = obj.detectors2do{ii};

                [this_detector, this_detector_name] = strtok(this_detector, ':');
                if( isempty(this_detector_name) )
                    this_detector_name = this_detector;
                else
                    this_detector_name = this_detector_name(2:end);
                end

                cprintf( 'Blue', [ 'Processing QRS detector ' this_detector_name ' @ ' Seconds2HMS(ECG_start_offset/ECG_header.freq) '\n' ] );

                %% perform QRS detection

                switch( this_detector )

                    case 'wavedet'
                    %% Wavedet delineation

                        try

                            obj.wavedet_config.setup.wavedet.QRS_detection_thr = repmat( obj.detection_threshold, 5, 1);

                            [position_multilead, positions_single_lead] = wavedet_interface(ECG, ECG_header, [], obj.lead_idx, obj.wavedet_config, ECG_sample_start_end_idx, ECG_start_offset, obj.progress_handle);

                            for jj = rowvec(obj.lead_idx)
                                % QRS detections in milliseconds
                                payload_out.(['wavedet_' obj.lead_names{jj} ]).time = colvec(positions_single_lead(jj).qrs);
                            end

                            % QRS detections in milliseconds
                            payload_out.wavedet_multilead.time = colvec(position_multilead.qrs);

                        catch aux_ME


                            strAux = sprintf('Wavedet failed in recording %s\n', ECG_header.recname);
                            strAux = sprintf('%s\n', strAux);

                            report = getReport(aux_ME);
                            fprintf(2, '%s\nError report:\n%s', strAux, report);

                        end

                    case 'pantom'
                        %% Pan-Tompkins detector

                        for jj = rowvec(obj.lead_idx)

                            try

                                aux_val = PeakDetection2(double(ECG(:,jj)), ECG_header.freq, [], [], [], obj.detection_threshold * 0.2, [], obj.progress_handle);

                                % filter and offset
                                aux_val = aux_val( aux_val >= ECG_sample_start_end_idx(1) &  aux_val <= ECG_sample_start_end_idx(2) ) + ECG_start_offset - 1;

                                % QRS detections in milliseconds
                                payload_out.(['pantom_' obj.lead_names{jj} ]).time = colvec(aux_val);

                            catch aux_ME


                                strAux = sprintf('Pan & Tompkins failed in recording %s lead %s\n', ECG_header.recname, ECG_header.desc(jj,:) );
                                strAux = sprintf('%s\n', strAux);

                                report = getReport(aux_ME);
                                fprintf(2, 'Error report:\n%s', report);

                            end

                        end

                    case { 'aristotle' 'gqrs' 'sqrs' 'wqrs' 'ecgpuwave' 'epltdqrs1' 'epltdqrs2'}
                    %% WFDB_comp_interface (sqrs, wqrs, aristotle, ecgpuwave)

                        [status, ~] = system([ obj.WFDB_cmd_prefix_str  this_detector ' -h' ]);

                        if( status ~= 0 )
                            disp_string_framed(2, sprintf('Could not execute "%s" QRS detector.', this_detector));  
                        else
                            % QRS detection.                        
                            for jj = rowvec(obj.lead_idx)

                                if( any(strcmpi( 'sqrs', this_detector ) ) )
                                    file_name_orig =  [obj.tmp_path_local ECG_header_WFDB.recname '.qrs' ];
                                    this_thrs = round(500 * obj.detection_threshold);
                                elseif( any(strcmpi( 'wqrs' , this_detector ) ) )
                                    file_name_orig =  [obj.tmp_path_local ECG_header_WFDB.recname '.wqrs' ];
                                    this_thrs = round(100 * obj.detection_threshold);
                                elseif( any(strcmpi( {'epltdqrs1' 'epltdqrs2'} , this_detector ) ) )
                                    file_name_orig =  [obj.tmp_path_local ECG_header_WFDB.recname '.epl' ];
                                    this_thrs = obj.detection_threshold;
                                else
                                    file_name_orig =  [obj.tmp_path_local ECG_header_WFDB.recname '.' this_detector num2str(jj) ];
                                    this_thrs = obj.detection_threshold;
                                end

                                try

                                    if( any(strcmpi( {'sqrs' 'wqrs' 'epltdqrs1' 'epltdqrs2'} , this_detector ) ) )
                                        % wqrs tiene una interface diferente
                                        [status, ~] = system([ obj.WFDB_cmd_prefix_str  this_detector ' -r ' ECG_header_WFDB.recname ' -s ' num2str(jj-1) ' -m ' num2str(this_thrs) ]);
                                        if( status ~= 0 );  disp_string_framed(2, sprintf('%s failed in recording %s lead %s', this_detector, ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) ) ); end

%                                     elseif( any(strcmpi( {'epltdqrs1' 'epltdqrs2'}, this_detector ) ) )
%                                         [status, ~] = system([ obj.WFDB_cmd_prefix_str  this_detector ' ' ECG_header_WFDB.recname ' ' num2str(jj-1)]);
%                                         if( status ~= 0 );  disp_string_framed(2, sprintf('%s failed in recording %s lead %s', this_detector, ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) ) ); end

                                    elseif( any(strcmpi( 'ecgpuwave', this_detector ) ) )
                                        [status, ~] = system([ obj.WFDB_cmd_prefix_str  this_detector ' -r ' ECG_header_WFDB.recname ' -a ' this_detector num2str(jj) ' -s ' num2str(jj-1)]);
                                        if( status ~= 0 );  disp_string_framed(2, sprintf('%s failed in recording %s lead %s', this_detector, ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) ) ); end

                                    elseif( any(strcmpi( 'aristotle', this_detector ) ) )
                                        [status, ~] = system([ obj.WFDB_cmd_prefix_str  this_detector ' -r ' ECG_header_WFDB.recname ' -o ' this_detector num2str(jj) ' -s ' num2str(jj-1) ' -G ' num2str(this_thrs) ]);
                                        if( status ~= 0 );  disp_string_framed(2, sprintf('%s failed in recording %s lead %s', this_detector, ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) ) ); end

                                    else

                                        if( strcmpi( 'gqrs', this_detector ) && exist(obj.gqrs_config_filename, 'file') )
                                            % using the configuration file to
                                            % post-process
                                            [status, ~] = system([ obj.WFDB_cmd_prefix_str 'gqrs -c ' obj.gqrs_config_filename ' -r ' ECG_header_WFDB.recname ' -s ' num2str(jj-1) ' -m ' num2str(this_thrs) ]);
                                            if( status == 0 )
                                                [status, ~] = system([ obj.WFDB_cmd_prefix_str 'gqpost -c ' obj.gqrs_config_filename ' -r ' ECG_header_WFDB.recname ' -o ' this_detector num2str(jj) ]);
                                                if( status ~= 0 )
                                                    disp_string_framed(2, sprintf('gqpost failed in recording %s lead %s', ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) ) );
                                                end
                                            else
                                                disp_string_framed(2, sprintf('gqrs failed in recording %s lead %s', ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) ) );
                                            end

                                            delete([obj.tmp_path_local ECG_header_WFDB.recname '.qrs' ]);
                                        else
                                            % run only WFDB compatible detector
                                            [status, ~] = system([ obj.WFDB_cmd_prefix_str  this_detector ' -r ' ECG_header_WFDB.recname ' -o ' this_detector num2str(jj) ' -s ' num2str(jj-1)]);
                                            if( status ~= 0 );  disp_string_framed(2, sprintf('%s failed in recording %s lead %s', this_detector, ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) ) ); end
                                        end

                                    end

                                catch aux_ME


                                    strAux = sprintf( '%s failed in recording %s lead %d\n', this_detector, ECG_header_WFDB.recname, jj);

                                    report = getReport(aux_ME);
                                    fprintf(2, '%s\nError report:\n%s', strAux, report);

                                end


                                if( status == 0  )

                                    if( exist(file_name_orig, 'file') )
                                        % reference comparison

                                        anns_test = [];                    

                                        try

                                            anns_test = readannot(file_name_orig);

                                            if( isempty(anns_test) )

                                                payload_out.([this_detector '_' obj.lead_names{jj} ]).time = [];

                                            else

                                                anns_test = AnnotationFilterConvert(anns_test, 'MIT', 'AAMI');

                                                if( obj.WFDBResampled )
                                                    %take sample locations
                                                    %to its original
                                                    %sampling rate
                                                    anns_test.time = round(anns_test.time * ECG_header.freq / ECG_header_WFDB.freq);
                                                end

                                                % filter and offset
                                                anns_test.time = anns_test.time(anns_test.time >= ECG_sample_start_end_idx(1) & anns_test.time <= ECG_sample_start_end_idx(2)) + ECG_start_offset - 1;

                                                payload_out.([this_detector '_' obj.lead_names{jj} ]).time = colvec(anns_test.time);

                                            end

                                        catch aux_ME

                                            if( strcmpi(aux_ME.identifier, 'MATLAB:nomem') )
                                                payload_out.([this_detector '_' obj.lead_names{jj} ]).time = [];
                                            else
                                                strAux = sprintf( '%s failed in recording %s lead %s\n', this_detector, ECG_header_WFDB.recname, ECG_header_WFDB.desc(jj,:) );

                                                report = getReport(aux_ME);
                                                error('ECGtask_QRS_detection:WFDB', '%s\nError report:\n%s', strAux, report);

                                            end

                                        end

                                        delete(file_name_orig);

                                    else

                                        payload_out.([this_detector '_' obj.lead_names{jj} ]).time = [];

                                    end

                                else

                                    payload_out.([this_detector '_' obj.lead_names{jj} ]).time = [];

                                    if( exist(file_name_orig, 'file') )
                                        delete(file_name_orig);
                                    end
                                end

                            end

                        end

                    case 'user'
                        %% user-defined QRS detector

                        try

                            if( exist(this_detector_name) == 2 )

%                                 ud_func_pointer = eval(['@' this_detector_name]);
                                ud_func_pointer = str2func(this_detector_name);

                                obj.progress_handle.checkpoint([ 'User defined function: ' this_detector_name ' @ ' Seconds2HMS(ECG_start_offset/ECG_header.freq) ])

                                ECG_header_aux = trim_ECG_header(ECG_header, obj.lead_idx);

                                [positions_single_lead, position_multilead] = ud_func_pointer( double(ECG(:,obj.lead_idx)), ECG_header_aux, obj.progress_handle, obj.payload);

                                for jj = 1:length(obj.lead_idx)

                                    aux_val = positions_single_lead{jj};
                                    % filter and offset
                                    aux_val = aux_val( aux_val >= ECG_sample_start_end_idx(1) &  aux_val <= ECG_sample_start_end_idx(2) ) + ECG_start_offset - 1;

                                    % QRS detections in milliseconds
                                    payload_out.([ this_detector_name '_' obj.lead_names{jj} ]).time = colvec(aux_val);

                                end

                                if( ~isempty(position_multilead) )
                                    % filter and offset
                                    position_multilead = position_multilead( position_multilead >= ECG_sample_start_end_idx(1) &  position_multilead <= ECG_sample_start_end_idx(2) ) + ECG_start_offset - 1;
                                    payload_out.([ this_detector_name '_multilead' ]).time = colvec(position_multilead);
                                end

                            else
                                disp_string_framed(2, sprintf('Function "%s" is not reachable in path.', this_detector_name));  
                                fprintf(1, 'Make sure that exist(%s) == 2\n',this_detector_name);
                            end

                        catch aux_ME

                            disp_string_framed(2, sprintf('Detector "%s" failed in recording %s', this_detector_name, ECG_header.recname ) );                                

                            report = getReport(aux_ME);
                            fprintf(2, 'Error report:\n%s', report);

                        end

                end

            end

            % Add QRS detections quality metrics, Names, etc.
            payload_out = calculateSeriesQuality(payload_out, ECG_header, [1 ECG_header.nsamp] + ECG_start_offset - 1 );
            
            % delete intermediate tmp files
            if(obj.bWFDBdetectors)
                delete([obj.tmp_path_local ECG_header.recname '.*' ]);
            end
            
            % calculate performance
            if( obj.CalculatePerformance )
                
                payload_out = CalculatePerformanceECGtaskQRSdet(payload_out, ECG_annotations, ECG_header, ECG_start_offset);

            end
            
        end
        
        function payload = Finish(obj, payload, ECG_header)
            
            if( isfield(payload, 'series_quality') && isfield(payload.series_quality, 'ratios') )
                payload.series_quality.ratios = mean(payload.series_quality.ratios, 2);
            end
                
        end
        
        function payload = Concatenate(obj, plA, plB)
            
            payload = ConcatenateQRSdetectionPayloads(obj, plA, plB);

        end

        %% property restriction functions
        
        function set.detectors(obj,x)
            
            if( (ischar(x) || iscellstr(x)) )
                x = cellstr(x);
                aux_val = colvec(intersect(obj.cQRSdetectors, x));
                aux_idx = find(cellfun(@(a)(~isempty(strfind(a, 'user:'))), x));
                
                if( isempty(aux_idx) && isempty(aux_val) )
                    warning('ECGtask_QRS_detection:BadArg', 'Invalid detectors.');
                else
                    obj.detectors = [aux_val; colvec(x(aux_idx)) ];
                end                    
                
            else
                warning('ECGtask_QRS_detection:BadArg', 'Invalid detectors.');
            end
        end
        
        function set.only_ECG_leads(obj,x)
            if( islogical(x) )
                obj.only_ECG_leads = x;
            else
                warning('ECGtask_QRS_detection:BadArg', 'Invalid lead configuration.');
            end
        end
        
        function set.lead_idx(obj,x)
            if( isnumeric(x) && all(x) > 0 )
                obj.lead_idx = x;
            else
                warning('ECGtask_QRS_detection:BadArg', 'Invalid lead indexes.');
            end
        end
        
        
        function set.gqrs_config_filename(obj,x)
            if( ischar(x) )
                
                if(exist(x, 'file'))
                    obj.gqrs_config_filename = x;
                else
                    warning('ECGtask_QRS_detection:BadArg', 'obj.gqrs_config_filename does not exist.');
                end
                
            else
                warning('ECGtask_QRS_detection:BadArg', 'obj.gqrs_config_filename must be a string.');
            end
        end
        
        function set.tmp_path_local(obj,x)
            if( ischar(x) )
                if(exist(x, 'dir'))
                    obj.tmp_path_local = x;
                else
                    if(mkdir(x))
                        obj.tmp_path_local = x;
                    else
                        warning('ECGtask_QRS_detection:BadArg', ['Could not create ' x ]);
                    end
                end
                
            else
                warning('ECGtask_QRS_detection:BadArg', 'tmp_path_local must be a string.');
            end
        end
        
        function set.tmp_path(obj,x)
            if( ischar(x) )
                if(exist(x, 'dir'))
                    obj.tmp_path = x;
                else
                    if(mkdir(x))
                        obj.tmp_path = x;
                    else
                        warning('ECGtask_QRS_detection:BadArg', ['Could not create ' x ]);
                    end
                end
                
            else
                warning('ECGtask_QRS_detection:BadArg', 'tmp_path_local must be a string.');
            end
        end
        
    end
    
    methods ( Access = private )
        
        
    end
    
end
