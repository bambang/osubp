# OSUBP

GPU-accelerated backprojection for synthetic aperture radar imaging developed at the Ohio State University, Department of Electrical and Computer Engineering in collaboration with the Department of Biomedical Informatics.

### Dive right in:
* the kernel: http://code.google.com/p/osubp/source/browse/PyCUDABackProjectionKernel.cu (read this)
* the PyCUDA setup: http://code.google.com/p/osubp/source/browse/pythonBP.py (run this)

### Upcoming:

* tiling
* double support
* real-time code generation in PyCUDA
* further analysis of the errors in output between Matlab and CUDA imagers
* non-Matlab data loaders, pre-processors, and visualization
* Matlab-only MEX interface

### Required packages:

* Python
* CUDA
* PyCUDA http://mathema.tician.de/software/pycuda
* mlabwrap http://mlabwrap.sourceforge.net/
* Matlab for data loading http://www.mathworks.com/products/matlab/

### Recommended:
* Sage http://sagemath.org/
* Raw phase history GOTCHA public datasets: GOTCHA Volumetric SAR Data Set and SAR-Based GMTI in Urban Environment Challenge Problem (please ignore "invalid SSL certificate" warning, and email SDMS asking them to fix it)

### References:

* AFRL bpBasic Matlab package: contact Mr LeRoy Gorham, AFRL: LeRoy.Gorham@WPAFB.AF.MIL, and see Gorham, et al., SAR image formation toolbox for MATLAB
* IEEE Radar Conference 2010 publication: Fasih, et al., GPU-accelerated synthetic aperture radar backprojection in CUDA
* PPAC @ IEEE Cluster 2009 publication: Hartley, et al., Investigating the Use of GPU-Accelerated Nodes for SAR Image Formation

### Contact:

Ahmed Fasih (last name dot 1 @osu.edu)

### Licence:

Ohio State-developed and -extended software's license:

COPYRIGHT (c) 2010, THE OHIO STATE UNIVERSITY
ALL RIGHTS RESERVED

Permission is granted to use, copy, create derivative works and
redistribute this software, associated documentation and such
derivative works (the "SOFTWARE") in source and object code form
provided that the above copyright notice, this grant of permission,
and the disclaimer below appear in all copies and derivatives made;
and provided that the SOFTWARE and any derivative works are made
available in source code form upon request; and provided that The Ohio
State University and authors of the SOFTWARE are acknowledged in any
publications reporting its use, and the name of The Ohio State
University or any of its officers, employees, students or board
members is not used in any advertising or publicity pertaining to the
use or distribution of the SOFTWARE without specific, written prior
authorization.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT REPRESENTATION FROM THE OHIO
STATE UNIVERSITY AS TO ITS FITNESS FOR ANY PURPOSE, AND WITHOUT
WARRANTY BY THE OHIO STATE UNIVERSITY OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. The Ohio State University has no obligation to
provide maintenance, support, updates, enhancements, or other
modifications. The Ohio State University shall not be liable for
compensatory or non-compensatory damages, including but not limited to
special, indirect, incidental, or consequential damages, with respect
to any claim arising out of or in connection with the use of the
SOFTWARE, even if it has been or is hereafter advised of the
possibility of such damages.  

(Other code, e.g., AFRL bpBasic codes, have separate licenses.)
