function color_map=green14
%
% Usage: color_map = green14
%
% Version X.X: unknown date
% Author:      unknown
%
% DESCRIPTION:   Colormap of WHITE-YELLOW-GREEN-BLUE-BLACK used
%         by MIT Lincoln Laboratory to display SAR images%
% Makes a 8-bit colormap good for viewing sar images. This is the
% replacement for the make_f14 script
%
% Dependencies: None
% Modifications:
color_map(80:191,1) = 0.9961*(0:111)'/111;color_map(192:256,1) = 0.9961*ones(65,1);
%%  construct green
%
color_map(3:256,2) = 0.9844*(0:253)'/253;
%%  construct blue
%
color_map(1:20,3) = 0.2344*(0:19)'/19;color_map(21:30,3) = 0.2344 + 0.039*(0:9)'/9;color_map(31:40,3) = 0.2734 - 0.039*(0:9)'/9;color_map(41:80,3) = 0.2344 - 0.2344*(0:39)'/39;color_map(180:245,3) = 0.9961*(0:65)'/65;color_map(246:256,3) = 0.9961*ones(11,1);
