classdef ECGtask_QRS_detections_post_process < ECGtask

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
%     [positions_single_lead, position_multilead] = your_ECG_delineator( ECG_matrix, ECG_header, progress_handle, payload_in);
% 
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
%          In the context of delineation, it is thought to be a user
%          corrected, or "gold quality" QRS location, in order to 
%          improve the wave delineation quality. If "payload_in" is a
%          struct, this function will automatically filter and time-shift
%          all QRS detection fields started with the string "corrected_".
%          For this purpose, QRScorrector task, automatically appends this
%          string to eachm anually reviewed QRS location series. 
% 
% the output of your function must be:
%    + positions_single_lead, a cell array size ECG_header.nsig with the
%          QRS sample locations found in each lead.
%    + position_multilead, a numeric vector with the QRS locations
%          calculated using multilead rules.
% 
% 
% Author: Mariano Llamedo Soria (llamedom at frba.utn.edu.ar)
% Version: 0.1 beta
% Birthdate  : 18/2/2013
% Last update: 18/2/2013
       
    properties(GetAccess = public, Constant)
        name = 'QRS_detections_post_process';
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
        
    end
    
    properties( Access = private )
        
        tmp_path_local
                
    end
    
    properties
        
        post_proc_func
        progress_handle
        payload
        tmp_path
        CalculatePerformance = false;
        bRecalcQualityAndPerformance = false;
        bRecalculateNewDetections = false;
        
    end
    
    methods
           
        function obj = ECGtask_QRS_detections_post_process (obj)
            
        end
        
        function Start(obj, ECG_header, ECG_annotations)
            
            obj.started = true;
            
        end
        
        function payload_out = Process(obj, ECG, ECG_start_offset, ECG_sample_start_end_idx, ECG_header, ECG_annotations, ECG_annotations_start_end_idx )
            
            payload_out = [];
            
            if( ~obj.started )
                obj.Start(ECG_header);
                if( ~obj.started )
                    cprintf('*[1,0.5,0]', 'Task %s unable to be started for %s.\n', obj.name, ECG_header.recname);
                    return
                end
            end
            
            post_proc_func_calc = [];

            if( obj.bRecalcQualityAndPerformance )
            
                % payload property is used in this task to input an external QRS
                % detector, or manually corrected detections. In this case,
                % with previous detections performed by this task.
                
                if( iscell(obj.payload) && length(obj.payload) > 1  )
                    
                    post_proc_struct = obj.payload{1};
                    QRSdet_struct = obj.payload{2};
                    
                    if( isstruct(post_proc_struct) && isstruct(QRSdet_struct) )
                    
                        if( isfield(QRSdet_struct, 'series_quality') && isfield(QRSdet_struct.series_quality, 'sampfreq' ) )
                            ann_sampfreq = QRSdet_struct.series_quality.sampfreq;
                        else
                            % assumed in the same sampling rate
                            ann_sampfreq = ECG_header.freq;
                        end

                        payload_out.series_quality.sampfreq = ECG_header.freq;

                        % ratio to convert annotations @ ann_sampfreq to ECG_header.freq
                        aux_sampfreq_ratio = ECG_header.freq / ann_sampfreq;
                        
                        
                        for ii = 1:size(post_proc_struct.series_quality.AnnNames,1)
                            aux_val = unique( round( post_proc_struct.(post_proc_struct.series_quality.AnnNames{ii,1}).(post_proc_struct.series_quality.AnnNames{ii,2}) * aux_sampfreq_ratio )) - ECG_start_offset + 1;
                            aux_val = aux_val( aux_val >= ECG_sample_start_end_idx(1) & aux_val <= ECG_sample_start_end_idx(2) );
                            % the previous detections must be shifted in
                            % order to mantain the references
                            aux_val = aux_val + ECG_start_offset - 1;
                            payload_out.(post_proc_struct.series_quality.AnnNames{ii,1}).(post_proc_struct.series_quality.AnnNames{ii,2}) = aux_val;
                        end

                        post_proc_func_calc = post_proc_struct.series_quality.AnnNames(:,1);
                        post_proc_func_calc = unique(cellfun( @(a)(a{1}), regexp(post_proc_func_calc, '(.+)_.+', 'tokens')));

                        AnnNames = QRSdet_struct.series_quality.AnnNames(:,1);

                        for fn = rowvec(AnnNames)
                            aux_val = unique( round( QRSdet_struct.(fn{1}).time * aux_sampfreq_ratio ) ) - ECG_start_offset + 1;
                            aux_val = aux_val( aux_val >= ECG_sample_start_end_idx(1) & aux_val <= ECG_sample_start_end_idx(2) );
                            QRSdet_struct.(fn{1}).time = aux_val;
                        end
                        
                        
                    else
                        cprintf('*[1,0.5,0]', 'Unable to parse payload struct for %s.\n', ECG_header.recname);
                        return                    
                    end
                    
                else
                    cprintf('*[1,0.5,0]', 'Unable to parse payload struct for %s.\n', ECG_header.recname);
                    return                    
                end                

            else

                % payload property is used in this task to input an external QRS
                % detector, or manually corrected detections.
                if( isstruct(obj.payload) )

                    QRSdet_struct = obj.payload;
                    
                    if( isfield(QRSdet_struct, 'series_quality') && isfield(QRSdet_struct.series_quality, 'sampfreq' ) )
                        ann_sampfreq = QRSdet_struct.series_quality.sampfreq;
                    else
                        % assumed in the same sampling rate
                        ann_sampfreq = ECG_header.freq;
                    end

                    % ratio to convert annotations @ ann_sampfreq to ECG_header.freq
                    aux_sampfreq_ratio = ECG_header.freq / ann_sampfreq;
                    
                    AnnNames = QRSdet_struct.series_quality.AnnNames(:,1);

                    for fn = rowvec(AnnNames)
                        aux_val = unique( round( QRSdet_struct.(fn{1}).time * aux_sampfreq_ratio ) ) - ECG_start_offset + 1;
                        aux_val = aux_val( aux_val >= ECG_sample_start_end_idx(1) & aux_val <= ECG_sample_start_end_idx(2) );
                        QRSdet_struct.(fn{1}).time = aux_val;
                    end

                else
                    cprintf('*[1,0.5,0]', 'Unable to parse payload struct for %s.\n', ECG_header.recname);
                    return
                end                

            end
            
            if( ~obj.bRecalculateNewDetections )
                % If recalculate only new detections, we exclude
                % detectors already processed.
                obj.post_proc_func = setdiff(obj.post_proc_func, post_proc_func_calc);
            end
            
            
            % Actual postproc functions
            for this_func = rowvec(obj.post_proc_func)
                
                try

                    obj.progress_handle.checkpoint([ 'User defined function: ' this_func{1}])

    %                 this_func_ptr = eval(['@' this_func]);
                    this_func_ptr = str2func(this_func{1});

                    this_payload = this_func_ptr( QRSdet_struct, ECG_header, ECG_sample_start_end_idx );

                    if( ~isempty(this_payload) )
                        for fn = rowvec(fieldnames(this_payload))
                            payload_out.(fn{1}) = this_payload.(fn{1});
                            aux_val = payload_out.(fn{1}).time + ECG_start_offset - 1;
                            payload_out.(fn{1}).time = aux_val;
                        end
                    end
                    
                catch aux_ME

                    disp_string_framed(2, sprintf('User-function "%s" failed in recording %s', this_func{1}, ECG_header.recname ) );                                

                    report = getReport(aux_ME);
                    fprintf(2, 'Error report:\n%s', report);

                end
                
            end

            obj.progress_handle.checkpoint('Adding quality metrics')

            % Add QRS detections quality metrics, Names, etc.
            payload_out = calculateSeriesQuality(payload_out, ECG_header, [1 ECG_header.nsamp] + ECG_start_offset - 1 );

            % calculate performance
            if( obj.CalculatePerformance )
                payload_out = CalculatePerformanceECGtaskQRSdet(payload_out, ECG_annotations, ECG_header, ECG_start_offset);
            end

            obj.progress_handle.checkpoint('Done')
            
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

        function set.post_proc_func(obj, x)
            
            if( isempty(x) )
                obj.post_proc_func = x;
            elseif( ischar(x) )
                
                if( exist(x) == 2 )
                    obj.post_proc_func = cellstr(x);
                else
                    disp_string_framed(2, sprintf('Function "%s" is not reachable in path.', x));  
                    fprintf(1, 'Make sure that exist(%s) == 2\n',x);
                end
                
            elseif( iscellstr(x) )

                if( any( cellfun( @(a)(exist(a)), x) == 2) )
                    obj.post_proc_func = x;
                else
                    disp_string_framed(2, sprintf('Function "%s" is not reachable in path.', x));  
                    fprintf(1, 'Make sure that exist(%s) == 2\n',x);
                end
                
            else
                warning('ECGtask_QRS_detections_post_process:BadArg', 'post_proc_func must be a string.');
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
                        warning('ECGtask_QRS_detections_post_process:BadArg', ['Could not create ' x ]);
                    end
                end
                
            else
                warning('ECGtask_QRS_detections_post_process:BadArg', 'tmp_path_local must be a string.');
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
                        warning('ECGtask_QRS_detections_post_process:BadArg', ['Could not create ' x ]);
                    end
                end
                
            else
                warning('ECGtask_QRS_detections_post_process:BadArg', 'tmp_path_local must be a string.');
            end
        end
        
    end
    
    methods ( Access = private )
        
        
    end
    
end
