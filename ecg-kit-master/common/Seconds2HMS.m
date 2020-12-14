%% (Internal) Create a string of hours mins and seconds based on data in seconds
%   
% 
%   [strRetVal, iHours, iMins, iSeconds, iMilliSeconds ] = Seconds2HMS(data, prec)
% 
% Arguments:
% 
%      + data: Time in seconds
% 
%      + prec: the number of decimals for the seconds decimals.
% 
% Output:
% 
%      + strRetVal: A string with the formatted time.
% 
%      + iHours, iMins, iSeconds, iMilliSeconds : Each part of the time.
% 
% Example:
% 
%         this_dur_str = Seconds2HMS( aux_dur, time_precision);
% 
% See also plot_ecg_strip
% 
% Author: Mariano Llamedo Soria llamedom@electron.frba.utn.edu.ar
% Version: 0.1 beta
% Last update: 14/5/2014
% Birthdate  : 21/4/2015
% Copyright 2008-2015
% 
function [strRetVal, iHours, iMins, iSeconds, iMilliSeconds ] = Seconds2HMS(data, prec, bFilenamefriendly )

if( nargin < 3 )
    % decimals of the seconds
    bFilenamefriendly = false;
end

if( nargin < 2 )
    % decimals of the seconds
    prec = 0;
end

sign_data = sign(data);
data = abs(data);

iDays = floor(data * 1 / 60 / 60 / 24);
iHours = floor(data * 1 / 60 / 60 - iDays * 24);
iMins = floor(data * 1 / 60 - iDays * 24 * 60 - iHours * 60 );
iSeconds = floor(data - iDays * 24 * 60  * 60 - iHours * 60  * 60 - iMins * 60);
iMilliSeconds = (data - iDays * 24 * 60  * 60 - iHours * 60  * 60 - iMins * 60 - iSeconds) * 1000;

if(bFilenamefriendly)
    seconds_str = 's';
else
    seconds_str = '"';
end

if(bFilenamefriendly)
    minutes_str = 'm ';
else
    minutes_str = ''' ';
end

ldata = length(data);
strRetVal = cell(ldata,1);

for ii = 1:ldata
    
    if( sign_data(ii) < 0 )
        strAux = '-'; 
    else
        strAux = [];
    end
    
    if( iDays(ii) > 0 )
        strAux = [ strAux num2str(iDays(ii)) 'd ' ]; 
    end
    
    if( iHours(ii) > 0 )
        strAux = [  strAux num2str(iHours(ii)) 'h ' ]; 
    end

    if( iMins(ii) > 0 )
        strAux = [  strAux num2str(iMins(ii)) minutes_str ]; 
    end
    
    if( iMilliSeconds(ii) > 0 || isempty(strAux) || iSeconds(ii) > 0 )
        
        if(prec > 0 )
            
            if( iSeconds(ii) > 0 )
                strAux = [  strAux sprintf( '%d.', iSeconds(ii) ) ]; 
                strAux2 = sprintf( '%03d', round(iMilliSeconds(ii)) ); 
                strAux = [  strAux strAux2(1:min(3,prec)) seconds_str ]; 
            else
                strAux2 = sprintf( [ '%3.' num2str(prec) 'f'  ], iMilliSeconds(ii) ); 
                aux_idx = find(strAux2 == '.');
                strAux = [  strAux strAux2(1:(aux_idx+prec)) ' ms' ]; 
            end
            
        else
            if( isempty(strAux) || iSeconds(ii) > 0 )
                strAux = [  strAux sprintf( ['%d' seconds_str], iSeconds(ii)) ]; 
            end
        end
        
    end
    
    strRetVal{ii} = strAux;
    
end

strRetVal = char(strRetVal);
