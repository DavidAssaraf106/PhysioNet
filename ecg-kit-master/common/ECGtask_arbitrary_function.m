classdef ECGtask_arbitrary_function < ECGtask

% ECGtask for ECGwrapper (for Matlab)
% ---------------------------------
% 
% Description:
% 
% Task to perform linear filtering in ECG signals
% 
% 
% Author: Mariano Llamedo Soria (llamedom at frba.utn.edu.ar)
% Version: 0.1 beta
% Birthdate  : 25/2/2015
% Last update: 25/2/2015
       
    properties(GetAccess = public, Constant)
        name = 'arbitrary_function';
        target_units = 'ADCu';
        doPayload = true;
    end

    properties( GetAccess = public, SetAccess = private)
        % if user = memory;
        % memory_constant is the fraction respect to user.MaxPossibleArrayBytes
        % which determines the maximum input data size.
        memory_constant = 0.5;
        
        started = false;

% to track the signal range over the whole signal.         
        range_min_max_tracking = [];
        
    end
    
    properties( Access = private, Constant)
    
    end
    
    properties( Access = private )
        interproc_data        
    end
    
    properties
        
        progress_handle 
        tmp_path
        only_ECG_leads = false;
        payload
        lead_idx = [];
        signal_payload = false;
        sampling_rate_out = [];
        function_pointer
        start_func_pointer = @default_start_function;
        finish_func_pointer = @default_finish_function;
        concate_func_pointer = @default_concatenate_function;
        
    end
    
    methods
           
        function obj = ECGtask_arbitrary_function(obj)

        end
        
        function Start(obj, ECG_header, ECG_annotations)

            if( obj.only_ECG_leads )
                obj.lead_idx = get_ECG_idx_from_header(ECG_header);
            else
                if( isempty(obj.lead_idx) )
    %               'all-signals'
                    obj.lead_idx = 1:ECG_header.nsig;
                else
                    obj.lead_idx = obj.lead_idx( obj.lead_idx >= 1 & obj.lead_idx <= ECG_header.nsig);
                end
            end
            
            if( isempty(obj.lead_idx) )
                cprintf('*[1,0.5,0]', 'Could not find any valid signal.\n');
                return
            end
            
            obj.range_min_max_tracking = [ repmat(realmax, 1, length(obj.lead_idx)); repmat(realmin, 1, length(obj.lead_idx)) ];

            if( isempty(obj.sampling_rate_out) )
                obj.sampling_rate_out = ECG_header.freq;
            end
            
            obj.interproc_data = [];
            
            obj.started = obj.start_func_pointer(obj, ECG_header);
            
        end
        
        function payload = Process(obj, ECG, ECG_start_offset, ECG_sample_start_end_idx, ECG_header, ECG_annotations, ECG_annotations_start_end_idx  )
            
            payload = [];

            if( ~obj.started )
                obj.Start(ECG_header);
                if( ~obj.started )
                    cprintf('*[1,0.5,0]', 'Task %s unable to be started for %s.\n', obj.name, ECG_header.recname);
                    return
                end
            end

            if( nargin(obj.function_pointer) == 1 )
                aux_payload = obj.function_pointer( double(ECG(:,obj.lead_idx)) );
            else
                ECG_header_aux = trim_ECG_header(ECG_header, obj.lead_idx);
                
                if( nargout(obj.function_pointer) == 2 )
                    [aux_payload, obj.interproc_data ] = obj.function_pointer( ECG(:,obj.lead_idx), ECG_header_aux, ECG_start_offset, obj.progress_handle, obj.payload, obj.interproc_data);
                else
                    aux_payload = obj.function_pointer( ECG(:,obj.lead_idx), ECG_header_aux, ECG_start_offset, obj.progress_handle, obj.payload);
                end                
            end
            
            if( obj.signal_payload )
                % trim the signal
                
                if( obj.sampling_rate_out ~= ECG_header.freq )
                    % freq change
                    kmultirate = obj.sampling_rate_out / ECG_header.freq;
                else
                    % no freq change
                    kmultirate = 1;
                end
                
                payload.result_signal = aux_payload(max(1,round(ECG_sample_start_end_idx(1)*kmultirate)):min(end, round(ECG_sample_start_end_idx(2)*kmultirate) ),:);
                
                cant_out_signals = size( payload.result_signal, 2 );
                
                obj.range_min_max_tracking = [ min( [obj.range_min_max_tracking(1,1:cant_out_signals); min(payload.result_signal) ] ); max([obj.range_min_max_tracking(2,1:cant_out_signals); max(payload.result_signal)]) ];
                
            else
                payload.result = aux_payload;
            end
            
        end
        
        function payload = Finish(obj, payload, ECG_header)
            
            payload = obj.finish_func_pointer( payload, ECG_header );
            
        end
        
        function payload = Concatenate(obj, plA, plB)

            payload = obj.concate_func_pointer(plA, plB);
            
        end

        function set.function_pointer(obj,x)
            
            if( strcmpi( class(x), 'function_handle' ) )
                obj.function_pointer = x;
                return
            end
            
            if( ischar(x) && exist(x) == 2 )
                obj.function_pointer = str2func(x);
            else
                warning('ECGtask_arbitrary_function:BadArg', 'Invalid function pointer.');
            end
            
        end
        
        function set.start_func_pointer(obj,x)
            
            if( strcmpi( class(x), 'function_handle' ) )
                obj.start_func_pointer = x;
                return
            end
            
            if( ischar(x) && exist(x) == 2 )
                obj.start_func_pointer = str2func(x);
            else
                warning('ECGtask_arbitrary_function:BadArg', 'Invalid function pointer.');
            end
            
        end
        
        function set.finish_func_pointer(obj,x)
            
            if( strcmpi( class(x), 'function_handle' ) )
                obj.finish_func_pointer = x;
                return
            end
            
            if( ischar(x) && exist(x) == 2 )
                obj.finish_func_pointer = str2func(x);
            else
                warning('ECGtask_arbitrary_function:BadArg', 'Invalid function pointer.');
            end
            
        end
        
        function set.concate_func_pointer(obj,x)
            
            if( strcmpi( class(x), 'function_handle' ) )
                obj.concate_func_pointer = x;
                return
            end
            
            if( ischar(x) && exist(x) == 2 )
                obj.concate_func_pointer = str2func(x);
            else
                warning('ECGtask_arbitrary_function:BadArg', 'Invalid function pointer.');
            end
            
        end
                
        function set.only_ECG_leads(obj,x)
            if( islogical(x) )
                obj.only_ECG_leads = x;
            else
                warning('ECGtask_arbitrary_function:BadArg', 'Argument must be a logical value.');
            end
        end
        
    end
    
   
    
end
