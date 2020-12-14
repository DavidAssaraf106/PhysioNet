%LABELD Find labels of classification dataset (perform crisp classification)
% 
%   LABELS = LABELD(Z)
%   LABELS = Z*LABELD
%   LABELS = LABELD(A,W)
%   LABELS = A*W*LABELD
%   LABELS = LABELD(Z,THRESH)
%   LABELS = Z*LABELD([],THRESH)
%   LABELS = LABELD(A,W,THRESH)
%   LABELS = A*W*LABELD([],THRESH)
%
% INPUT
%		Z        Classification dataset, or
%		A,W      Dataset and classifier mapping
%   THRESH   Rejection threshold
%
% OUTPUT
%		LABELS	List of labels
%
% DESCRIPTION 
% Returns the labels of the classification dataset Z (typically the result
% of a mapping or classification A*W). For each object in Z (i.e. each row)
% the feature label or class label (i.e. the column label) of the maximum 
% column value is returned.
%
% Effectively, this performs the classification. It can also be considered
% as a conversion from soft labels Z to crisp labels.
%
% When the parameter THRESH is supplied, then all objects which
% classifier output falls below this value are rejected. The returned
% label is then NaN or a string with spaces (depending if the labels are
% numeric or string). Because the output of the classifier is used, it
% is recommended to convert the output to a posterior prob. output using
% CLASSC.                                    (David Tax,  27-12-2004)
% 
% SEE ALSO
% MAPPINGS, DATASETS, TESTC, PLOTC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function labels = labeld(a,w,thresh)

		% Add the possibility to reject objects for which the posterior is
	% too low:
	if (nargin < 3)
		thresh = [];
	end
	if (nargin == 2) & isa(w,'double') % we did something like labeld(z,0.3)
		thresh = w;
		w = [];
	end
	if (nargin < 2)
		w = [];
	end

	if (nargin == 0) | isempty(a)

		% Untrained mapping.
		labels = prmapping(mfilename,'fixed',{thresh});

	elseif isempty(w)
		
		if (isdatafile(a)) % datafile needs to process objects separately
			labels = cell(size(a,1),1);
	    next = 1;
	    while next > 0
		    [b,next,J] = readdatafile(a,next);
				labs = feval(mfilename,b,[],thresh);
				for i=1:length(J)
					labels{J(i)} = labs(i,:);
				end
			end
			if isstr(labels{1})
				labels = char(labels);
			else
				labels = cell2mat(labels);
			end
			return
		end
		
		

		% In a classified dataset, the feature labels contain the output
		% of the classifier.
		[m,k] = size(a); featlist = getfeatlab(a);
		Jrej = []; % as a start, we don't reject objects

		if (k == 1)
			% If there is one output, assume it's a 2-class discriminant: 
			% decision boundary = 0. 
			J = 2 - (double(a) >= 0); 
			if ~isempty(thresh)
				warning('Inproper thresholding of the 2-class dataset, please use classc.');
			end
		else
			% Otherwise, pick the column containing the maximum output.
			[dummy,J] = max(+a,[],2);
			% Reject the objects which have posteriors lower than the
			% threshold
			if ~isempty(thresh)
				Jrej = find(dummy<thresh);
			end
		end
		labels = featlist(J,:);
		% Take care for the rejected objects:
		if ~isempty(Jrej)
			if isa(featlist,'double')
				labels(Jrej) = NaN;
			elseif isa(featlist,'char')
% 				labels(Jrej,:) = repmat(' ',size(featlist(1,:)) ); % Trick!:-)
				labels(Jrej,:) = repmat(' ', length(Jrej), size(labels,2) ); % Trick!:-)
			else
				error('The featlist of A is confusing. Cannot make a reject label');
			end
		end
	else

		% Just construct classified dataset and call again.
		labels = feval(mfilename,a*w,thresh);

	end

return

