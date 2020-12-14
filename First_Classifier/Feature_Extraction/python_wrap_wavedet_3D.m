function [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12] = python_wrap_wavedet_3D(input_directory, ecg, filename, list_lead, qrs_peak)

    num_leads = length(list_lead);
    disp(ecg);
    disp(length(ecg))
    out_1 = 0;
    out_2 = 0;
    out_3 = 0;
    out_4 = 0;
    out_5 = 0;
    out_6 = 0;
    out_7 = 0;
    out_8 = 0;
    out_9 = 0;
    out_10 = 0;
    out_11 = 0;
    out_12 = 0;
    for i=1:num_leads
        ecg_lead = ecg{i};
        ecg_here = horzcat(ecg_lead{:});
        ech_here = transpose(ecg_here)
        heasig = struct("nsig",1,"freq",500,"nsamp",length(ecg_here));
        if(length(qrs_peak)==0)
            [position,~,~] = wavedet_3D(ecg_here, [], heasig, []);
        end
        if(length(qrs_peak)>0)
            [position,~,~] = wavedet_3D(ecg_here, double(qrs_peak(i,:)), heasig, []);
        end
        if(i==1)
            out_1 = position;
        end
        if(i==2)
            out_2 = position;
        end
        if(i==3)
            out_3 = position;
        end
        if(i==4)
            out_4 = position;
        end
        if(i==5)
            out_5 = position;
        end
        if(i==6)
            out_6 = position;
        end
        if(i==7)
            out_7 = position;
        end
        if(i==8)
            out_8 = position;
        end
        if(i==9)
            out_9 = position;
        end
        if(i==10)
            out_10 = position;

        end
        if(i==11)
            out_11 = position;
        end
        if(i==12)
            out_12 = position;
        end

    end

end




