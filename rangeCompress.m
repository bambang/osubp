function [data] = rangeCompress(data)

has = @(s) isfield(data,  s);
get = @(s) getfield(data, s);

if has('taper_flag') && (get('taper_flag'))
    if has('taper_window')
        win2d = get('taper_window');
    else
        [h w] = size(data.phdata);
        win2d = taylorwin(h, 5, -35) * taylorwin(w, 5, -35)';
    end
else
    win2d = 1;
end

phase_ramp = exp(-1j*4*pi/data.clight*data.minF.*data.R0);

data.upsampled_range_profiles = ifftshift(...
    ifft(win2d .* data.phdata, data.Nfft,1), 1) .* repmat(phase_ramp,data.Nfft, 1);

end
