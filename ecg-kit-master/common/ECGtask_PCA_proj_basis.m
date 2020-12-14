classdef ECGtask_PCA_proj_basis < ECGtask

% ECGtask for ECGwrapper (for Matlab)
% ---------------------------------
% 
% Description:
% 
% Abstract class for defining ECGtask interface
% 
% 
% Author: Mariano Llamedo Soria (llamedom at frba.utn.edu.ar)
% Version: 0.1 beta
% Birthdate  : 18/2/2013
% Last update: 18/2/2013
       
    properties(GetAccess = public, Constant)
        name = 'PCA_proj_basis';
        target_units = 'ADCu';
        doPayload = false;
    end

    properties( GetAccess = public, SetAccess = private)
        % if user = memory;
        % memory_constant is the fraction respect to user.MaxPossibleArrayBytes
        % which determines the maximum input data size.
        memory_constant = 0.3;
        
        started = false;
        
    end
    
    properties( Access = private, Constant)
    
        %sample 10 minutes from the whole recording.
        time_sample = 10 * 60; %seconds
        time_heartbeat_window = 2; %seconds around the heartbeat
    end
    
    properties( Access = private )
        ECG_slices
        QRS_sample_idx
        wavelet_filters
        some_window
        halfwin_samples
    end
    
    properties
        progress_handle 
        cant_QRS_locations
        autovec
        tmp_path
        
        
    end
    
    methods
           
        function obj = ECGtask_PCA_proj_basis(obj)
            obj.ECG_slices = [];
        end
        
        function Start(obj, ECG_header, ECG_annotations)

            % ECG cell for grouping QRS complex slices
            obj.ECG_slices = [];
                                    
            % Sample which QRS complexes will consider into de PCA basis
            % calculation
            cant_QRS_sample = round(obj.time_sample / obj.time_heartbeat_window);

            cant_QRS = length(ECG_annotations.time);
            
            obj.QRS_sample_idx = sort(randsample(cant_QRS, min(cant_QRS,cant_QRS_sample)));
            
            obj.halfwin_samples = round( obj.time_heartbeat_window/2*ECG_header.freq);
            
            obj.some_window = colvec(blackman(2*obj.halfwin_samples+1));

            obj.started = true;
            
        end
        
        function payload = Process(obj, ECG, ECG_start_offset, ECG_sample_start_end_idx, ECG_header, ECG_annotations, ECG_annotations_start_end_idx )
            
            % this object doesn't generate any payload
            payload = [];

            if( ~obj.started )
                obj.Start(ECG_header);
                if( ~obj.started )
                    cprintf('*[1,0.5,0]', 'Task %s unable to be started for %s.\n', obj.name, ECG_header.recname);
                    return
                end
            end
            
            this_iter_QRS_sample_idx = find(obj.QRS_sample_idx >= ECG_annotations_start_end_idx(1) & obj.QRS_sample_idx <= ECG_annotations_start_end_idx(2));
            
            if( ~isempty(this_iter_QRS_sample_idx) )
                
                % change the reference to this iteration
                this_iter_QRS_sample_idx = obj.QRS_sample_idx(this_iter_QRS_sample_idx) - ECG_annotations_start_end_idx(1) + 1;
                
                QRS_locations = ECG_annotations.time;

                this_iter_ECG_size = size(ECG,1);

                ECG = int16(round(BaselineWanderRemovalSplines( double(ECG), QRS_locations, ECG_header.freq)));

                aux_idx = arrayfun(@(a)( max(1, QRS_locations(a) - obj.halfwin_samples): ...
                                         min( this_iter_ECG_size, QRS_locations(a) + obj.halfwin_samples)) , ...
                                   colvec(this_iter_QRS_sample_idx), 'UniformOutput', false);

                aux_idx3 = 1:(2*obj.halfwin_samples+1);
                aux_idx2 = arrayfun(@(a)( aux_idx3( ...
                                         (max(1, QRS_locations(a) - obj.halfwin_samples): ...
                                         min( this_iter_ECG_size, QRS_locations(a) + obj.halfwin_samples)) ...
                                         - max(1, QRS_locations(a) - obj.halfwin_samples) + 1 ) ), ...
                                   colvec(this_iter_QRS_sample_idx), 'UniformOutput', false);

                obj.ECG_slices = [obj.ECG_slices; cellfun(@(a,b)( int16(round(bsxfun(@times, double(ECG(a,:)), obj.some_window(b) ))) ), aux_idx, aux_idx2, 'UniformOutput', false)];
            
            end
            
        end
        
        function payload = Finish(obj, payload, ECG_header)
            
            ECG = cell2mat(obj.ECG_slices);

            % Wavelet transform calculation
            wtECG = int16(round(qs_wt(double(ECG), 4, ECG_header.freq, obj.wavelet_filters)));

            % calculate PCA in wt scale 4 of the ECG
            WTecg_cov = obj.my_mcdcov( double(wtECG) );
            [obj.autovec autoval] = eig(WTecg_cov); 
            autoval = diag(autoval);
            [~, autoval_idx] = sort(autoval, 'descend');
            obj.autovec = obj.autovec(:,autoval_idx);   
            
        end
        
        function payload = Concatenate(obj, plA, plB)
            
            payload = [];
            % not implemented
            
        end

    end
    
    methods ( Access = private )
        
        function cov_mat = my_mcdcov(obj, x)
            
            result = mcdcov(x, 'plots', 0);
            cov_mat = result.cov;
            
        end
        
    end
    
    
end
