%% (Internal) Set random properties a line handle
%
%     all_proerties = set_rand_linespec(line_hdl, mrk, linestyle, col_spec, markerSize)
% 
% 
% Arguments:
% 
%   + line_hdl: The handle
% 
%   + mrk: marker cellstring to choose
% 
%   + linestyle: linestyle cellstring to choose
% 
%   + col_spec: colors to choose
% 
%   + markerSize: marker size to choose
% 
% Output:
% 
%   + all_proerties: Properties set.
% 
% Example:
% 
% 
% See also plot_ecg_strip
% 
% Author: Mariano Llamedo Soria (llamedom at frba.utn.edu.ar)
% Version: 0.1 beta
% Birthdate: 17/12/2010
% Last update: 17/12/2010
% Copyright 2008-2015
% 
function all_proerties = set_rand_linespec(line_hdl, mrk, linestyle, col_spec, markerSize)

all_proerties = [];

if(~ishandle(line_hdl))
    return;
end

cant_line_hdl = length(line_hdl);

if( nargin < 2 )
    mrk={'+','o','*','.','x','s','d','^','v','<','>','p','h'};
else
    if( ischar(mrk) )
        mrk = cellstr(mrk);
    elseif( ~iscellstr(mrk) )
        mrk={'+','o','*','.','x','s','d','^','v','<','>','p','h'};
    end    
end

if( nargin < 3 )
    linestyle = {'-','--',':','-.'};
else
    if( ischar(linestyle) )
        linestyle = cellstr(linestyle);
    elseif( ~iscellstr(linestyle) )
        linestyle = {'-','--',':','-.'};
    end    
end

if( nargin < 4 || isempty(col_spec) || (~isnumeric(col_spec) &&  (size(col_spec,2) == 3) ) )
    col_spec = jet;
end

if( nargin < 5 )
    markerSize = {5};
else
    if( isnumeric(markerSize) )
        markerSize = num2cell(markerSize);
    elseif( ~iscell(markerSize) )
        markerSize = {5};
    end    
end

mrk = colvec(mrk);
linestyle = colvec(linestyle);
markerSize = colvec(markerSize);

colspec_mf = col_spec(randsample(size(col_spec,1),cant_line_hdl, true),: );
colspec_me = col_spec(randsample(size(col_spec,1),cant_line_hdl, true),: );
colspec_line = col_spec(randsample(size(col_spec,1),cant_line_hdl, true),: );

markers_idx = colvec(randsample(length(mrk), cant_line_hdl, true ));
linestyles_idx = colvec(randsample(length(linestyle), cant_line_hdl, true ));
markerSize_idx = colvec(randsample(length(markerSize), cant_line_hdl, true ));

set(line_hdl,   {'Marker'}, mrk(markers_idx), ... 
                {'MarkerSize'}, markerSize(markerSize_idx), ...
                {'MarkerFaceColor'}, num2cell(colspec_mf,2), ...
                {'MarkerEdgeColor'}, num2cell(colspec_me,2), ...
                {'LineStyle'},linestyle(linestyles_idx) , ...
                {'Color'}, num2cell(colspec_line,2));

if( nargout > 0)
    all_proerties = {mrk linestyle markerSize colspec_mf colspec_me colspec_line markers_idx linestyles_idx markerSize_idx};
end
