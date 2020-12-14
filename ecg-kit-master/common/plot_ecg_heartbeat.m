%% (Internal) obsolete, use plot_ecg_strip
% This function is used by the ECGtask related with QRS detection.
%   
% Example
% 
% Arguments:
% 
% Output:
% 
% See also plot_ecg_strip
% 
% Author: Mariano Llamedo Soria llamedom@electron.frba.utn.edu.ar
% Version: 0.1 beta
% Last update: 14/5/2014
% Birthdate  : 23/4/2013
% Copyright 2008-2015

function [ECG_hdl, ECG_limits] = plot_ecg_heartbeat(ECG, lead_idx, QRS_locations, ECG_start_idx, QRS_start_idx, cant_qrs, heasig, filtro, axes_hdl, limits )

ECG_hdl = [];
ECG_limits = [];

if( nargin < 8 || isempty(limits) )
    bFixAxes = false;
else
    bFixAxes = true;
end


if( nargin < 7 )
    filtro = [];
end

if( nargin < 8 || isempty(axes_hdl) )
    axes_hdl = gca;
end

% cla(axes_hdl);
% axes(axes_hdl);

axes(axes_hdl)
fig_hdl = gcf;


lECG = size(ECG,1);
cant_sig = length(lead_idx);

if( isempty(QRS_locations) )
    QRS_start_idx = [];
    cant_qrs = [];
end

if( iscell(QRS_locations) )
    other_QRS_locations = QRS_locations(2:end);
    QRS_locations = QRS_locations{1};
else
    other_QRS_locations = [];
end
    
cant_QRS_locations = length(QRS_locations);

QRS_start_idx = min(cant_QRS_locations, QRS_start_idx);

if( isfield(heasig, 'btime') )
    aux_val = datevec(datenum(heasig.btime, 'HH:MM:SS'));
    base_start_time = round((aux_val(4) * 60 * 60 + aux_val(5) * 60 + aux_val(6)) * heasig.freq);
else
    base_start_time = 1;
end

if( isempty(QRS_locations) ) 
    start_idx = 1;
    end_idx = 10*heasig.freq;
else
    
    if( (QRS_start_idx-cant_qrs) < 1 )
        start_idx = max(1, (QRS_locations(QRS_start_idx) -ECG_start_idx + 1) - 5 * heasig.freq );
    else
        
        % avoid gaps between marks
        aux_start = max(1, QRS_start_idx-cant_qrs);
        aux_idx = aux_start:QRS_start_idx;
        aux_marks = QRS_locations(aux_idx);
        valid_marks_idx = find(~isnan(aux_marks));
        aux_diffs = diff(aux_marks);
        median_diff = median(aux_diffs);
        gap_idx = find( aux_diffs > 5*median_diff , 1, 'last' );
        
        if( isempty(gap_idx) )
            
            if( isempty(valid_marks_idx) )
                start_idx = 1;
            else
                if( length(valid_marks_idx) < cant_qrs )
                    start_idx = max(1, (QRS_locations(aux_idx(valid_marks_idx(1)))-ECG_start_idx + 1)- 5 * heasig.freq );
                else
                    start_idx = max(1, (QRS_locations(aux_idx(valid_marks_idx(1)))-ECG_start_idx + 1) );
                end
                
            end
            
        else
            start_idx = max(1, (aux_marks(gap_idx+1)-ECG_start_idx + 1) - 5 * heasig.freq);
        end
            
    end
    
    if( QRS_start_idx >= (cant_QRS_locations-cant_qrs) )
        end_idx = min(lECG, (QRS_locations(QRS_start_idx) -ECG_start_idx + 1) + 5 * heasig.freq );
    else
        
        aux_end = min(cant_QRS_locations, QRS_start_idx+cant_qrs);
        aux_idx = QRS_start_idx:aux_end;
        aux_marks = QRS_locations(aux_idx);
        valid_marks_idx = find(~isnan(aux_marks));
        aux_diffs = diff(aux_marks);
        median_diff = median(aux_diffs);
        gap_idx = find( 5*median_diff < aux_diffs, 1, 'last' );
        
        if( isempty(gap_idx) )
            if( isempty(valid_marks_idx) )
                end_idx = lECG;
            else
                if( length(valid_marks_idx) < cant_qrs )
                    end_idx = min(lECG, (QRS_locations(aux_idx(valid_marks_idx(end)))-ECG_start_idx + 1) + 5 * heasig.freq );
                else
                    end_idx = min(lECG, (QRS_locations(aux_idx(valid_marks_idx(end)))-ECG_start_idx + 1) );
                end
            end
        else
            end_idx = min(lECG, (aux_marks(gap_idx)-ECG_start_idx + 1) + 5 * heasig.freq);
        end
        
    end
    
    
end
aux_idx = start_idx:end_idx;


if( ~isempty(filtro) && ~isempty(aux_idx) )
    % zero phase filtering of the ECG signals only.
    ECG_idx = get_ECG_idx_from_header(heasig);
    
    aux_lead_idx = intersect(lead_idx, ECG_idx);
    
    if( isempty(aux_lead_idx) )
        if( isempty(ECG_idx) )
            cprintf('[1,0.5,0]', disp_option_enumeration( 'No filter was applied since no ECG leads found. Lead description found:', cellstr(heasig.desc) ) )
            fprintf(1, '\n')
        end
    else
        aux_idx2 = max(1, aux_idx(1) - 1 * heasig.freq  ):min(lECG, aux_idx(end) + 1 * heasig.freq );
        orig_class = class(ECG);
        aux_val = filter(filtro, double(flipud(ECG(aux_idx2,aux_lead_idx))) );
% round works bad with floating point.
%         ECG(aux_idx2,aux_lead_idx) = cast( round(filter(filtro, flipud(aux_val))), orig_class);
        ECG(aux_idx2,aux_lead_idx) = cast( (filter(filtro, flipud(aux_val))), orig_class);
    end
    
end

if( cant_sig > 1 )
    max_values = max(ECG(aux_idx,lead_idx));
    min_values = min(ECG(aux_idx,lead_idx));
    aux_ranges = max_values - min_values;
    aux_sig = bsxfun( @minus, double(ECG(aux_idx,lead_idx)), mean(ECG(aux_idx,lead_idx)));
    aux_sig = bsxfun( @times, aux_sig, double(max(aux_ranges))./double(aux_ranges) );
    ecg_max = max(max(aux_sig));
    ecg_min = min(min(aux_sig));
    
else
    aux_sig = ECG(aux_idx,lead_idx);
    ecg_max = max(aux_sig);
    ecg_min = min(aux_sig);
end

ecg_range = ecg_max - ecg_min;
k_range = 0.05;
ecg_lims = [ ecg_min + k_range*ecg_range ecg_max - k_range*ecg_range ];

if( isempty(aux_idx) )
    qrs_ploted =[];
else
    qrs_ploted = find(QRS_locations >= (aux_idx(1) + ECG_start_idx - 1) & QRS_locations <= (aux_idx(end) + ECG_start_idx - 1) );
end

cla(axes_hdl);

ColorOrder = my_colormap( 12 );

set(axes_hdl, 'ColorOrder', ColorOrder);

if( ~bFixAxes )
    limits = prctile(colvec(aux_sig), [5 95]);

    aux_val = max(diff(limits), 0.05*limits(1) );
    aux_val = 1.1*aux_val;
    limits(1) = max(min(colvec(aux_sig)), limits(1) - aux_val);
    limits(2) = min(max(colvec(aux_sig)), limits(2) + aux_val);
end

set(axes_hdl, 'Ylim', limits );
ECG_limits = limits;

hold(axes_hdl, 'on')

ECG_hdl = colvec(arrayfun(@(sig_idx, col_idx)(plot(axes_hdl, aux_idx, aux_sig(:,sig_idx), 'Color', ColorOrder(col_idx,:) )), 1:length(lead_idx), rowvec(lead_idx), 'UniformOutput', false ));

% ECG_hdl = [ECG_hdl; colvec( arrayfun( @(a,b)( plot(axes_hdl, a, b , 'r--' )), repmat(rowvec(QRS_locations(qrs_ploted) - ECG_start_idx + 1 ),2,1), [ repmat(ecg_max, 1,length(qrs_ploted)) ; zeros(1,length(qrs_ploted)) ], 'UniformOutput', false ) ) ];
ECG_hdl = [ECG_hdl; colvec( cellfun( @(a,b)( plot(axes_hdl, a, b, ':r' )), mat2cell( repmat(rowvec(QRS_locations(qrs_ploted) - ECG_start_idx + 1 ),2,1), 2, ones(length(qrs_ploted),1) ), mat2cell ( [ repmat(ecg_max, 1,length(qrs_ploted)) ; repmat(ecg_min, 1,length(qrs_ploted))], 2, ones(length(qrs_ploted),1) ), 'UniformOutput', false) ) ];

aux_yrange = get(axes_hdl, 'Ylim');

ytext_loc = double(aux_yrange(2) - 0.05*ecg_range);

if( ~isempty(qrs_ploted) )
    aux_hdl2 = [];
    for ii = rowvec(qrs_ploted)
        aux_hdl = text( QRS_locations(ii) - ECG_start_idx + 1, ytext_loc, num2str(ii), 'BackgroundColor', [1 1 1], 'EdgeColor', ColorOrder(1,:), 'Margin', 2 );
    end

    uistack(aux_hdl2, 'bottom');

    aux_text_pos = get(aux_hdl, 'Extent');

    aux_val = unique(QRS_locations(qrs_ploted) - ECG_start_idx + 1 );
    set(axes_hdl, 'XTick', rowvec(aux_val) );
    if( length(aux_val) > 1 )
        aux_str = cellstr(Seconds2HMS(colvec(aux_val(1) + base_start_time - 1 + ECG_start_idx - 1)*1/heasig.freq));
        if( length(aux_val) > 2 )
            aux_str = [aux_str; cellstr(Seconds2HMS(colvec(aux_val(2:end-1))*1/heasig.freq,3)) ];
        end
        aux_str = [aux_str; cellstr(Seconds2HMS(colvec(aux_val(end) + base_start_time - 1 + ECG_start_idx - 1)*1/heasig.freq)) ];
        set(axes_hdl, 'XTickLabel', aux_str );
    else
        aux_str = cellstr(Seconds2HMS(colvec(aux_val + base_start_time - 1 + ECG_start_idx - 1)*1/heasig.freq));
        set(axes_hdl, 'XTickLabel', aux_str );
    end
end


if( isempty(other_QRS_locations) )
    
    ytext_loc = aux_yrange(2);
    
else
    
    jj = 2;
    x_offset = aux_text_pos(3);
    
    for this_QRS_location = rowvec(other_QRS_locations)
        this_QRS_locations = this_QRS_location{1};
        qrs_ploted = find(this_QRS_locations >= aux_idx(1) & this_QRS_locations <= aux_idx(end) );
        ECG_hdl = [ECG_hdl; colvec( cellfun( @(a,b)( plot(axes_hdl, a, b, 'LineStyle', ':', 'Color', ColorOrder(jj,:) )), mat2cell( repmat(rowvec(this_QRS_locations(qrs_ploted)),2,1), 2, ones(length(qrs_ploted),1) ), mat2cell ( [ repmat(ecg_max, 1,length(qrs_ploted)) ; zeros(1,length(qrs_ploted))], 2, ones(length(qrs_ploted),1) ), 'UniformOutput', false) ) ];
        for ii = rowvec(qrs_ploted)
            text( this_QRS_locations(ii) + x_offset, ytext_loc, num2str(ii), 'FontSize', 7, 'BackgroundColor', [1 1 1], 'EdgeColor', ColorOrder(jj,:) );
        end
        
        x_offset = x_offset + aux_text_pos(3);
%         ytext_loc = ytext_loc+aux_text_pos(4)/3;
        jj = jj + 1;
        
    end
    
end

aux_yrange = [aux_yrange(1) ytext_loc ];

set(axes_hdl, 'Ylim', aux_yrange);
set(axes_hdl, 'Box', 'off');

hold(axes_hdl, 'off');

