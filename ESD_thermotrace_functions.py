'''
This file contains all the needed functions to run the ESD_thermotrace simulations
Last edited by A. Madella on 16th February 2021
'''

import numpy as np                                 # library for arrays
import pandas as pd                                # library for tables
import geopandas as gpd                            # library for georeferenced tables
import warnings                                    # python warnings module
from collections import OrderedDict                # ordered dictionary objects
import matplotlib.pyplot as plt                    # plotting library
import matplotlib.gridspec as gs                   # library to make gridded subplots
from matplotlib.pyplot import cm                   # colour management liibrary
import seaborn as sns                              # pretty statistical plotting library
import scipy.interpolate as intr                   # interpolation functions
from scipy.stats import ks_2samp                   # Kolmogorov-Smirnov test
from sklearn.linear_model import LinearRegression  # linear regression function
import utm                                         # conversion from and to UTM coordinates
from os import mkdir, path                         # operating system utilities
from sklearn.manifold import MDS                   # multi dimensional scaling package
import ScientificColourMaps6 as scm6               # load colour maps by F. Crameri

############################################################################################################

def read_input_file(filename):
    '''
    function to read the input textfile
    the single argument must have extension .txt
    it returns a list of all the lines
    where input parameters are located in the last element of each splitted line
    '''
    fid = open(filename, 'r')
    fid = np.array(str(fid.read()).split('\n'))
    fid = fid[[len(i)!=0 for i in fid]]
    fid = fid[[i[0]!='#' for i in fid]]
    fid = [line.split(':') for line in fid]
    fid = [line[0].split()+line[1].split() for line in fid]

    # make integers where needed
    for line in fid:
        for i in np.arange(len(line)):
            try:
                line[i] = int(line[i])
            except:
                pass

    # make empty lists for e_maps and detrital data, if not given
    for i in [3,7]:
        if len(fid[i])<3:
            fid[i].append([])
        elif len(fid[i])>3:
            fid[i] = fid[i][:2]+[fid[i][2:]]
        elif type(fid[i][2]) != list:
            fid[i][2] = fid[i][2:]

    # make True/False for example_scenarios flag
    ex_scen = fid[9]
    if ex_scen[-1].capitalize() == 'True':
        ex_scen[-1] = True
    elif ex_scen[-1].capitalize() == 'False':
        ex_scen[-1] = False

    # change other ungiven parameters to None
    for line in [fid[4]]+[fid[6]]+fid[11:]:
        if len(line)<3:
            line.append(None)

    return fid

############################################################################################################

class DEM:
    '''
    class for digital elevation models, with all needed DEM attributes
    '''
    def __init__(self, name):
        self.name = name

    def from_ascii(self, filename):
        '''
        Function to read an ASCII raster, most commonly exported from ArcGIS or QGIS.
        The first 6 lines of the file must start like this:

        ncols
        nrows
        xllcorner
        yllcorner
        cellsize
        NODATA_value

        followed by the raster values separated by spaces
        '''

        # read DEM text file
        fid = open(filename, 'r')

        # make a table of the dem info and convert values to suitable data types (integer, float)
        dem_info = [fid.readline().split() for i in range(6)]
        dem_info = np.array(dem_info).transpose()
        dem_info = dict([(k,v) for k,v in zip(dem_info[0],dem_info[1])])
        dtypes = (int,int,float,float,float,float)
        for i,f in zip(dem_info,dtypes):
            dem_info[i]=f(dem_info[i])

        # get the dem data as a list of strings
        dem_ls_of_str = [fid.readline().split() for i in range(dem_info['nrows'])]

        # then convert all strings to floats
        dem = np.array([[float(i) for i in dem_ls_of_str[j]] for j in range(dem_info['nrows'])])
        if dem.shape != (dem_info['nrows'], dem_info['ncols']):
            warnings.warn('something went wrong while parsing the DEM\nnrows and/or ncols do not match the original input')

        # change NODATA_value to np.nan, unless it equals 0
        if dem_info['NODATA_value'] != 0:
            dem[dem==dem_info['NODATA_value']]=np.nan
            dem_info['NODATA_value']=np.nan

        self.z = dem
        self.info_dict = dem_info
        # specify the figure's geographical extent in lat,lon
        self.xllcorner = dem_info['xllcorner']
        self.yllcorner = dem_info['yllcorner']
        self.ncols = dem_info['ncols']
        self.nrows = dem_info['nrows']
        self.cellsize = dem_info['cellsize']
        self.nodata = dem_info['NODATA_value']
        self.extent84 = (self.xllcorner, self.xllcorner+self.ncols*self.cellsize,
                         self.yllcorner, self.yllcorner+self.nrows*self.cellsize)

        # build coordinate grids and arrays
        # convert llcorner and urcorner coordinates to utm and define extentUTM
        self.xyll = utm.from_latlon(self.extent84[2], self.extent84[0]) #force_zone_number=19
        self.xyur = utm.from_latlon(self.extent84[3], self.extent84[1]) #force_zone_number=19
        self.extentUTM = (self.xyll[0], self.xyur[0], self.xyll[1], self.xyur[1])

        # make easting and northing vectors
        Xi = np.linspace(self.xyll[0], self.xyur[0], self.ncols)
        Yi = np.linspace(self.xyll[1], self.xyur[1], self.nrows)
        self.xi, yi = np.meshgrid(Xi,Yi)
        self.yi = yi[::-1] ################ flipped row order for latitude to decrease from top

        # 1d vectors, needed for linear interpolation
        self.xi_1d = self.xi.reshape(dem.size)
        self.yi_1d = self.yi.reshape(dem.size)
        self.zi_1d = dem.reshape(dem.size)

    def info(self):
        '''
        function to print details of imported DEM class, except nodata value
        '''
        print('\nMETADATA OF '+self.name+'\n')
        print('xllcorner = {}'.format(self.xllcorner))
        print('yllcorner = {}'.format(self.yllcorner))
        print('ncols = {}'.format(self.ncols))
        print('nrows = {}'.format(self.nrows))
        print('cellsize [km] = {}'.format(self.cellsize))
        print('cellsize [m] = ~{}'.format(int(np.around(self.cellsize*110000,1)/10)*10))
        print('min value = {}'.format(np.nanmin(self.z)))
        print('max value = {}'.format(np.nanmax(self.z)))
        print('NODATA_value = {}'.format(self.nodata))

    def resample(self, res, xyll=None, xyur=None, extent84=None, method='nearest'):
        '''
        Method to resample the input rasters to desired resolution .
        By default it uses a nearest neighbour algorithm
        res = resolution in meters
        xyll = (x,y) coordinates of lower left corner of target extent of the resampled dem
        xyur = (x,y) coordinates of upper right corner of target extent of the resampled dem
        extent84 = (left,right,bottom,top) in lon-lat, of the target extent of the resampled dem
        method --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
        '''

        if xyll == None:
            xyll=self.xyll
        if xyur == None:
            xyur=self.xyur
        if extent84 == None:
            extent84=self.extent84

        # first make resampled easting and northing vectors
        Xi_res = np.arange(xyll[0], xyur[0]+res, res)
        Yi_res = np.arange(xyll[1], xyur[1]+res, res)
        self.xi_res, yi_res = np.meshgrid(Xi_res, Yi_res)
        self.yi_res = yi_res[::-1] ################# flipped row order for latitude to decrease from top
        self.xi_res_1d = self.xi_res.reshape(self.xi_res.size)
        self.yi_res_1d = self.yi_res.reshape(self.xi_res.size)

        # also make lon-lat vectors at same resolution
        Xi_res84 = np.linspace(extent84[0], extent84[1], self.xi_res.shape[1])
        Yi_res84 = np.linspace(extent84[2], extent84[3], self.xi_res.shape[0])
        self.xi_res84, yi_res84 = np.meshgrid(Xi_res84,Yi_res84)
        self.yi_res84 = yi_res84[::-1] ############## flipped row order for latitude to decrease from top

        # resample by interpolating at new grid nodes with resolution "res"
        # input_coords are organized in a 2D array, with columns representing x,y
        input_coords = np.concatenate(([self.xi_1d],[self.yi_1d])).transpose()
        # resampled coords are organized in a 2D array, with columns representing x,y
        self.res_coords = np.concatenate(([self.xi_res_1d],[self.yi_res_1d])).transpose()
        # Now resample: the 'values' variable refers to the known elevations of the input dem
        self.zi_res_1d = intr.griddata(points=input_coords, values=self.zi_1d, xi=self.res_coords, method=method)
        self.zi_res = self.zi_res_1d.reshape(self.xi_res.shape) # reshape from 1D to 2D

############################################################################################################

def import_dem(dem_filename, ipf):
    '''
    imports any dem and makes a class instance out if 
    '''
    try:
        key = dem_filename[:dem_filename.find('.')]
        dem = DEM(key)
        dem.from_ascii(ipf+'/'+dem_filename)
        dem.info()
        return dem
    except:
        return None

############################################################################################################

def import_e_maps(e_map_filenames, f_map_filename, ipf):
    '''
    imports the specified erosion map files, the fertility map and makes a dictionary out of them
    '''
    e_maps = {}
    # fill the dictionary, if erosion maps are given
    try:
        for M in e_map_filenames+[f_map_filename]:
            key = M[:M.find('.')]
            e_maps[key] = DEM(key)
            e_maps[key].from_ascii(ipf+'/'+M)
            e_maps[key].info()
        return e_maps
    except:
        return e_maps
############################################################################################################

def import_age_map(age_map_filename, age_map_u_filename, ipf, interp_method):
    '''
    it first checks for the interpolation method,
    if 'imp' it imports the specified maps of bedrock age and related uncertainty
    '''
    if interp_method == 'imp':
        age_map = DEM('bedrock age')
        age_map.from_ascii(ipf+'/'+age_map_filename)
        age_map.info()
        age_map_u = DEM('bedrock age uncertainty')
        age_map_u.from_ascii(ipf+'/'+age_map_u_filename)
        age_map_u.info()
        return age_map, age_map_u
    else:
        return None, None

############################################################################################################

def import_bedrock_data(path):
    '''
    function to import the bedrock data
    path - string of path ending with either .xlsx or .csv extension
    the linked table should have the following header:
    latitude, longitude, elevation, age, age_u, zero_age_depth
        - latitude and longitude must be WGS1984 values
        - age and age_u inform the age and related uncertainty in million years
        - zero_age_depth informs the depth at which the thermochronometric ages are reset.
        - this latter column can be blank, in which case the depth is assigned to equal elevation-5000 m
        - both elevation and zero_age_depth are in meters above sea level
    '''
    try:
        bd = pd.read_excel(path)
    except:
        bd = pd.read_csv(path)

    bd.sort_values(by='elevation',inplace=True)
    z = bd.elevation.values
    try:
        z0age = bd.zero_age_depth.values
        if not np.array(z0age==z0age).all(): # check for Nans
            z0age = z-5000
            warnings.warn('The column "zero_age_depth" contains Nans\n all zero age depths will be set to z-5000m')
    except:
        z0age = z-5000

    a, u, lat, lon = bd.age.values, bd.age_u.values, bd.latitude.values, bd.longitude.values

    # convert from geographic to projected coordinates, not to overestimate elevation during interpolation.
    x_utm, y_utm = np.array([]), np.array([]) # preallocate arrays
    for i,v in enumerate(lat):
        xy_utm = utm.from_latlon(lat[i], lon[i]) #force_zone_number=19
        x_utm, y_utm = np.append(x_utm, xy_utm[0]), np.append(y_utm, xy_utm[1])

    # add points at -5000 m below sample elevation (zero cooling age depth) to arrays
    # such that each x,y location has a double with age=0 and elevation=z-5000
    # alternatively use the zero_age_depth column, if present.
    xx_utm, yy_utm, zz, aa = x_utm, y_utm, z, a # double letters indicate doubled vectors thereon
    for i,v in enumerate(z):
        xx_utm, yy_utm = np.append(xx_utm, x_utm[i]), np.append(yy_utm, y_utm[i])
        zz, aa = np.append(zz, z0age[i]), np.append(aa, 1e-9)

    return bd, z, a, u, lat, lon, x_utm, y_utm, xx_utm, yy_utm, zz, aa

############################################################################################################

def plot_input_data(dem, e_maps, ws_outline, interp_method, bd,
                    dem_cmap, age_cmap, ero_cmap, age_map, age_map_u, saveas):
    '''
    function to plot input dem, bedrock samples, catchment outline, erosion maps.
    dem - instance of the class DEM, the imported digital elevation model
    e_maps - the dictionary of erosion maps
    ws_outline - watershed outline, geopandas.DataFrame
    interp_method - the chosen method of interpolation (see jupyter notebook)
    bd - lan, lot, age, age_sd table, (pandas.DataFrame)
    dem_cmap - any matplotlib colormap
    age_cmap - any matplotlib colormap
    ero_cmap - any matplotlib colormap
    saveas - path to filename to save the figure
    age_map - if interp_method is 'imp', the imported age_map
    age_map_u - if interp_method is 'imp', the imported age_map_u
    '''

    # make figure, gridspec and axes
    # you can edit the parameter "figsize" if the aspect ratio doesn't fit
    if len(e_maps)>0:
        fig = plt.figure(figsize=(15,15*2*dem.nrows/dem.ncols))
        gspec = gs.GridSpec(2,len(e_maps.keys()),figure=fig)
    else:
        fig = plt.figure(figsize=(15,15*dem.nrows/dem.ncols))
        gspec = gs.GridSpec(1,1,figure=fig)

    # first row of plot
    ax1 = fig.add_subplot(gspec[0,:])
    im = ax1.imshow(dem.z, origin='upper', cmap=dem_cmap, extent=dem.extent84) # raster plot
    poly = ws_outline.plot(edgecolor='k',facecolor='None',ax=ax1) # polygon plot
    ax1.set(aspect='equal', xlabel='longitude', ylabel='latitude',
            xlim=(dem.extent84[0],dem.extent84[1]), ylim=(dem.extent84[2],dem.extent84[3]))
    cb1 = fig.colorbar(im)
    cb1.set_label('elevation[m]')

    # scatter plot if map of bedrock data was not imported
    if interp_method != 'imp':
        ax1.set_title('Input DEM and bedrock samples',pad=10, fontdict=dict(weight='bold'))
        sct = ax1.scatter(bd.longitude.values, bd.latitude.values, c=bd.age.values, cmap=age_cmap, edgecolor='w')
        cb2 = fig.colorbar(sct)
        cb2.set_label('age [My]')
    else:
        ax1.set_title('Input DEM',pad=10, fontdict=dict(weight='bold'))

    # plot imported erosional maps, if present
    if len(e_maps)>0:
        count = 0
        row2 = []
        max_e = max([np.nanmax(m.z) for k,m in e_maps.items()]) # max erosion rate imported
        for k,i in e_maps.items():
            count+=1
            ax = fig.add_subplot(gspec[1,count-1])
            row2.append(ax)
            im2= ax.imshow(dem.z, origin='upper', cmap=dem_cmap, extent=dem.extent84)
            em = ax.imshow(i.z, origin='upper', cmap=ero_cmap, extent=dem.extent84, alpha=0.5, vmin=0, vmax=max_e)
            ws_outline.plot(edgecolor='k',facecolor='None',ax=ax)
            ax.set_title('Erosion Map: '+k, fontdict=dict(weight='bold'))
            if count == gspec.get_geometry()[1]:
                cb3 = fig.colorbar(em,orientation='horizontal',ax=row2)
                cb3.set_label('erosional/fertility weight')

    fig.savefig(saveas, dpi=200) # save figure

    if interp_method == 'imp': # plot for 'imp' method
        if age_map != None:
            fig = plt.figure(figsize=(15,15))
            gspec = gs.GridSpec(2,1,figure=fig)
            ax1 = fig.add_subplot(gspec[0])
            ax2 = fig.add_subplot(gspec[1])
            im1 = ax1.imshow(age_map.z, extent=dem.extent84, cmap=age_cmap)
            ax1.set_title('Imported map of bedrock age',pad=10, fontdict=dict(weight='bold'))
            im2 = ax2.imshow(age_map_u.z, extent=dem.extent84)
            ax2.set_title('Imported map of bedrock age error',pad=10, fontdict=dict(weight='bold'))
            cb1 = fig.colorbar(im1, ax=ax1)
            cb1.set_label('age [My]')
            cb2 = fig.colorbar(im2, ax=ax2)
            cb2.set_label('relative uncertainty [%]')
            fig.savefig(saveas, dpi=200) # save figure
        else:
            warnings.warn('you have selected the interpolation method "imp", but I cannot find an age map')

############################################################################################################

def dist3D(xyz1, xyz2):
    '''
    Calculates the distance between two points in 3D.
    xyz1 - list or tuple of x,y,z coords for first point
    xyz2 - list or tuple of x,y,z coords for second point
    '''
    return np.sqrt((xyz1[0]-xyz2[0])**2+(xyz1[1]-xyz2[1])**2+(xyz1[2]-xyz2[2])**2)

############################################################################################################

def extrapolation(gdop, gdopx, gdopy, gdopz, data, datax, datay, dataz, ext_rad):
    '''
    Extrapolates data within wanted radius
    using an inverse distance weighted linear regression of the available data points
    input arguments:
    gdop: griddata output,
          a 1D-array that contains the interpolated data
          as well as the nans that you want to replace
    gdopx
    gdopy
    gdopz: 1D coordinate arrays of griddata output

    data: 1D-array of known ages

    datax
    datay
    dataz: 1D coordinate arrays of known data points

    ext_rad: maximum extrapolation radius in meters
    '''

    # select nans from the griddata output
    nans, nansx, nansy, nansz = gdop[gdop!=gdop], gdopx[gdop!=gdop], gdopy[gdop!=gdop], gdopz[gdop!=gdop]

    # This is the workflow of the extrapolation function:
    # for each of the nans:
    # calculate inverse distance from NaN to all samples, drop samples too far away
    # multiply inverse distances by related age and store in a [1 x M] vector of weighted values
    # summate and divide by M
    for i in np.arange(nans.size):
        # make array of ages divided by distance and number of data points
        dists = np.array([dist3D((nansx[i],nansy[i],nansz[i]), (datax[j],datay[j],dataz[j])) for j in np.arange(data.size)])
        dists1 = dists[dists < ext_rad] # do not consider points farther than extra_rad
        if dists1.size > 0:
            data1 = data[dists < ext_rad] # select related ages
            dataz1 = dataz[dists < ext_rad]
            dists1 = 1/dists1 # invert distances
            # make linear regression based on age-elevation from data points within extra_rad
            f_z = LinearRegression().fit(dataz1.reshape((-1,1)), data1, sample_weight=dists1)
            nans[i] = f_z.intercept_+f_z.coef_*nansz[i]
        else:
            nans[i] = np.nan

    # now substitute nans with extrapolated values
    gdop[gdop!=gdop] = nans
    return gdop

############################################################################################################

def plot_resDEM_and_age_map(dem, res, age_interp_map, bd, ws_outline, interp_method, dem_cmap, age_cmap, saveas):
    '''
    plots resampled dem and interpolated age map

    dem - instance of the class DEM, the imported digital elevation model
    res - resolution in meters
    age_interp_map - 2d-array of the interpolated age map
    interp_method - the chosen method of interpolation (see jupyter notebook)
    bd - lan, lot, age, age_sd table, (pandas.DataFrame)
    ws_outline - watershed outline, geopandas.DataFrame
    dem_cmap - any matplotlib colormap
    age_cmap - any matplotlib colormap
    saveas - path to filename to save the figure

    '''
    fig = plt.figure(figsize=(15,15))
    gspec = gs.GridSpec(2,1,figure=fig)

    # upper panel, resampled DEM
    ax1 = fig.add_subplot(gspec[0])
    im1 = ax1.imshow(dem.zi_res, origin='upper', extent=dem.extent84, cmap=dem_cmap)
    ws_outline.plot(edgecolor='k',facecolor='None',ax=ax1)
    ax1.set(aspect='equal', ylabel='latitude', xticks=[])
    ax1.set_title('Resampled DEM ('+str(res)+'m resolution)', pad=10, fontdict=dict(weight='bold'))
    cb = fig.colorbar(im1)
    cb.set_label('elevation [m a.s.l.]')

    # lower panel, bedrock surface age map
    ax2 = fig.add_subplot(gspec[1])
    im2 = ax2.imshow(age_interp_map, origin='upper', extent=dem.extent84, cmap=age_cmap, alpha=1)
    ws_outline.plot(edgecolor='w',facecolor='None',ax=ax2)
    ax2.set(aspect='equal', xlabel='longitude', ylabel='latitude')
    ax2.set_title('Interpolated surface bedrock age', pad=10, fontdict=dict(weight='bold'))
    # plot samples and ages if interpolated map was not imported at the beginning
    vmin = min(np.nanmin(age_interp_map),bd.age.min())
    vmax = max(np.nanmax(age_interp_map),bd.age.max())
    if interp_method != 'imp':
        for ax in [ax1,ax2]:
            ax.scatter(x=bd.longitude, y=bd.latitude, c=bd.age, cmap=age_cmap, vmin=vmin, vmax=vmax, edgecolor='w')
            ax.set(xlim=(dem.extent84[0],dem.extent84[1]), ylim=(dem.extent84[2],dem.extent84[3]))

    m = cm.ScalarMappable(cmap=age_cmap)
    m.set_clim(vmin, vmax)
    cb = fig.colorbar(m)
    cb.set_label('age [My]')
    fig.savefig(saveas, dpi=200) # save fig

############################################################################################################

def make_errormap(dem, res, z, a, u, aa, xx_utm, yy_utm, zz, x_utm, y_utm, interp_method, ext_rad):
    '''
    dem - instance of the class DEM, the imported digital elevation model
    res - resolution in meters
    z - array of elevations from bedrock data
    a - array of ages from bedrock data
    u - array of age uncertainties from bedrock data
    aa - array of surface ages and zero ages
    xx_utm - array x_utm repeated twice
    yy_utm - array y_utm repeated twice
    zz - array of bedrock samples elevations and zero age depths
    x_utm - array of longitudes utm
    y_utm - array of latitudes utm
    interp_method - chosen interpolation method (input parameter)
    ext_rad - chosen radius of extrapolation in meters (input parameter)
    '''
    # pre-allocate a vector with as many elements as bedrock samples
    error_interp = np.zeros(a.size)
    # Bootstrap: for each bedrock sample...
    # 1) exclude the related point from input data,
    # 2) calculate an interpolated surface age at its location,
    # 3) save the difference to the known measured age.
    for i in np.arange(a.size):
        a_boot,x_boot,y_boot,z_boot = np.delete(aa, i), np.delete(xx_utm, i), np.delete(yy_utm, i), np.delete(zz, i)
        # interpolate error with chosen method
        if interp_method == 'rbf':
            rbfi = intr.Rbf(x_boot, y_boot, z_boot, a_boot, function='linear')
            a_int = rbfi(xx_utm[i], yy_utm[i], zz[i]) # get interpolated age of excluded sample
        else:
            pts1 = np.concatenate(([x_boot],[y_boot],[z_boot])).transpose() # data without i sample
            pos1 = np.concatenate(([xx_utm[i]], [yy_utm[i]], [zz[i]])) # coordinates of i sample
            a_int = intr.griddata(points=pts1, values=a_boot, xi=pos1)[0] #interpolated age of i sample
            if a_int != a_int and interp_method == 'ext':
                # extrapolate from remaining samples within ext_rad
                # make array of ages divided by distance and number of data points
                dists = np.array([dist3D((xx_utm[i],yy_utm[i],zz[i]), (x_boot[j],y_boot[j],z_boot[j])) for j in np.arange(a_boot.size)])
                dists1 = dists[dists < ext_rad] # do not consider points farther than extra_rad
                if dists1.size > 0:
                    data1 = a_boot[dists < ext_rad] # select ages with same index
                    dataz1 = z_boot[dists < ext_rad] # select elevations with same index
                    dists1 = 1/dists1 # invert distances
                    # make linear regression based on age-elevation from data points within extra_rad
                    f_z = LinearRegression().fit(dataz1.reshape((-1,1)), data1, sample_weight=dists1)
                    a_int = f_z.intercept_+f_z.coef_*zz[i]
                else:
                    a_int = np.nan
        error_interp[i] = abs(aa[i]-a_int)/aa[i]*100

    error_total = np.sqrt(error_interp**2+(u/a*100)**2) # calculate sqrt of the square error_interp + square age_sd
    error_total[error_total!=error_total] = np.nanmean(error_total) # substitute nans at edges with mean error

    # Now make the map by spatially interpolating known error points using chosen interpolation method
    if interp_method == 'rbf':
        rbfi = intr.Rbf(x_utm, y_utm, error_total, function='linear') # not considering elevation here
        age_interp_error_map = rbfi(dem.xi_res, dem.yi_res) # get interpolated age of excluded sample
    else:
        pts_err = np.concatenate(([x_utm],[y_utm])).transpose()
        age_interp_error = intr.griddata(points=pts_err, values=error_total, xi=dem.res_coords)
        # preallocate array
        age_interp_error_map = np.ones(dem.zi_res.shape)
        for i in np.arange(age_interp_error.size):
            x_ind = int(np.rint((dem.xi_res_1d[i]-dem.xyll[0])/res))
            y_ind = int(np.rint((dem.yi_res_1d[i]-dem.xyll[1])/res))
            # assign nan if out of bounds
            if dem.zi_res[y_ind][x_ind] != dem.zi_res[y_ind][x_ind]:
                age_interp_error_map[y_ind][x_ind] = np.nan
            else:
                age_interp_error_map[y_ind][x_ind] = age_interp_error[i]
        age_interp_error_map = age_interp_error_map[::-1] # flipped to have correct latitude

        if interp_method == 'ext': # call the extrapolation function if needed
            extra_error = extrapolation(age_interp_error_map.reshape(age_interp_error_map.size),
                                        dem.xi_res_1d, dem.yi_res_1d, dem.zi_res_1d,
                                        error_total, x_utm, y_utm, z, ext_rad)
            age_interp_error_map = extra_error.reshape(dem.zi_res.shape)

    return error_interp, age_interp_error_map

############################################################################################################

def make_errormap_zln(dem, z, a, u, n=1000):
    '''
    for the method 'zln', it iteratively determines the regression error as a function of elevation only
    and assign each elevation the corresponding error
    '''
    error_interp = np.zeros(z.size)
    for i in np.arange(z.size):
        A_iterations = []
        for j in np.arange(n):
            A1 = np.array([np.random.normal(AGE,UNC) for AGE,UNC in zip(a,u)])
            reg1 = LinearRegression().fit(z.reshape(-1,1),A1.reshape(-1,1))
            A_iterations.append(reg1.intercept_+reg1.coef_*z[i])
        # multiply standard deviation by 1.96 to approximate 95% confidence
        error_interp[i] = 1.96*np.std(A_iterations)/a[i]*100
    error_total = error_interp # total error is the same, because x,y components don't matter in this method
    age_interp_error = np.interp(dem.zi_res_1d, z, error_total)
    age_interp_error_map = age_interp_error.reshape(dem.zi_res.shape)
    return error_interp, age_interp_error_map

############################################################################################################

def plot_linreg(R2, reg0, z, a, u, error_interp, saveas):
    '''
    plots the linear regression used in case of import_method 'zln'
    R2 - coefficient of regression
    reg0 - regression model
    z - array of elevations from bedrock data
    a - array of ages from bedrock data
    u - array of age uncertainties from bedrock data
    error_interp - array of interpolation error, one element per bedrock data point
    saveas - path to filename to save the figure
    '''
    if R2 < 0.7:
        warnings.warn('The scatter of your age-elevation data is rather high (R^2 = '+str(R2)+')\nYou might want to use a different interpolation method')

    fig,ax = plt.subplots(1,1,figsize=(10,6))
    a_new = reg0.intercept_+reg0.coef_[0]*z
    ax.fill_between(x=z, y1=a_new-error_interp/100*a_new, y2=a_new+error_interp/100*a_new,color='k',alpha=0.3,
                   label='95% confidence from 1000 iterations')
    ax.plot(z, a_new, label='_nolegend_')
    ax.errorbar(z, a, yerr=u, fmt='ok', label='_nolegend_')
    ax.set(xlabel='elevation [m]', ylabel='age [My]')
    ax.set_title('Linear regression of bedrock age-elevation, R^2 = '+str(R2), pad=10, fontdict=dict(weight='bold'))
    ax.legend()
    fig.savefig(saveas, dpi=200) # save fig

############################################################################################################

def plot_error_map(age_interp_error_map, error_interp, bd, ws_outline, interp_method, err_cmap, extent, saveas):
    '''
    plots map of interpolation error

    age_interp_error_map - 2d-array, map of the total percent error sqrt(interpolation^2 + analytical^2)
    error_interp - 1d array, same length as bd, informs error only from interpolation
    bd - lan, lot, age, age_sd table, (pandas.DataFrame)
    ws_outline - watershed outline, geopandas.DataFrame
    err_cmap - any matplotlib colormap
    extent - extent of the input DEM in WGS84 (left,right,bottom,top)
    saveas - path to filename to save the figure
    '''
    fig,ax = plt.subplots(1,1,figsize=(20,7))
    vmax = max(np.nanmax(age_interp_error_map),np.nanmax(error_interp))
    im = ax.imshow(age_interp_error_map, origin='upper', extent=extent, cmap=err_cmap, vmin=0, vmax=vmax)
    ws_outline.plot(edgecolor='w',facecolor='None',ax=ax)
    ax.set(aspect='equal', xlabel='longitude', ylabel='latitude',
           xlim=(extent[0],extent[1]), ylim=(extent[2],extent[3]))
    ax.set_title('Map of Interpolation Error', pad=10, fontdict=dict(weight='bold'))

    if interp_method == 'imp' or interp_method == 'zln':
        cb = fig.colorbar(im)
        cb.set_label('bedrock age error [%]')
    else:
        cb = fig.colorbar(im)
        cb.set_label('raster: total error [%]')
        sct = ax.scatter(x=bd.longitude, y=bd.latitude, c=error_interp, edgecolor='w', cmap=err_cmap, vmin=0, vmax=vmax)
        cb1 = fig.colorbar(sct)
        cb1.set_label('scatter: interpolation error [%]')
    fig.savefig(saveas, dpi=200) # save fig

############################################################################################################

def clip_to_ws(raster, shp_filename, extent, input_folder, output_folder):
    '''
    This function clips the input raster to the watershed shapefile,
    so that all raster cells can be used to predict detrital distributions
    - raster: 2D np.array
    - shp_filename: string, filename of the watershed shapefile
    - extent: tuple or list, extent of the raster in wgs1984 (west, east, south, north)
    '''
    import fiona
    import rasterio
    from rasterio.plot import show
    from rasterio.mask import mask
    from shapely.geometry import Polygon

    # calculate x and y cellsize in degrees
    xsize = np.abs(extent[0]-extent[1])/raster.shape[1]
    ysize = np.abs(extent[2]-extent[3])/raster.shape[0]

    # define rasterio transform function
    transform = rasterio.transform.from_origin(extent[0], extent[3], xsize, ysize)

    # define coordinate reference system to wgs1984
    crs = rasterio.crs.CRS.from_epsg(4326) # wgs1984: 4326

    # make new raster file from input, necessary to use rasterio's functions, and define the metadata
    src = rasterio.open(output_folder+'/temp/raster.tif', 'w', driver='GTiff',
                        height = raster.shape[0], width = raster.shape[1],
                        count = 1, dtype = str(raster.dtype),
                        crs = crs, transform=transform)

    src.write(raster, 1) # write and close the new tif file
    src.close()

    # get watershed polygon vertices
    with fiona.open(input_folder+'/'+shp_filename,'r') as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # read the raster and make masking information
    with rasterio.open(output_folder+'/temp/raster.tif','r') as src:
        out_image, out_transform = mask(src, shapes, nodata=np.nan)
        out_meta = src.meta

    # update the metadata accordingly
    out_meta.update({"driver": "GTiff", "height": out_image.shape[1],
                     "width": out_image.shape[2], "transform": out_transform})

    # write the clipped raster
    with rasterio.open(output_folder+'/temp/raster_clipped.tif', 'w', **out_meta) as dest:
        dest.write(out_image)
    return rasterio.open(output_folder+'/temp/raster_clipped.tif').read(1)

############################################################################################################

def plot_clipped_age_map(dem, age_interp_map_clp, ws_outline, bd, interp_method, dem_cmap, age_cmap, saveas):
    '''
    plots the age map, clipped to the watershed outline, overlaid on dem.

    dem - instance of the class DEM, the imported digital elevation model
    age_interp_map_clp - 2d-array of the interpolated age map, clipped to watershed
    ws_outline - watershed outline, geopandas.DataFrame
    bd - lan, lot, age, age_sd table, (pandas.DataFrame)
    interp_method - the chosen method of interpolation (see jupyter notebook)
    dem_cmap - any matplotlib colormap
    age_cmap - any matplotlib colormap
    saveas - path to filename to save the figure
    '''
    fig, ax = plt.subplots(figsize=(15,7))
    ax.imshow(dem.zi_res, extent=dem.extent84, origin='upper', cmap=dem_cmap)
    im = ax.imshow(age_interp_map_clp, extent=dem.extent84, origin='upper', cmap=age_cmap)
    ws_outline.plot(edgecolor='w',facecolor='None',ax=ax)
    ax.set(aspect='equal', xlabel='longitude', ylabel='latitude',
           xlim=(dem.extent84[0],dem.extent84[1]), ylim=(dem.extent84[2],dem.extent84[3]))
    ax.set_title('Clipped Age Map', pad=10, fontdict=dict(weight='bold'))
    cb = fig.colorbar(im)
    cb.set_label('age [My]')

    if interp_method != 'imp':
        ax.scatter(x=bd.longitude, y=bd.latitude, c=bd.age, edgecolor='w', cmap=age_cmap,
                   vmin=np.nanmin(age_interp_map_clp), vmax=np.nanmax(age_interp_map_clp))

    fig.savefig(saveas, dpi=200) # save fig

############################################################################################################
    
def make_watershed_table(dflt_grids, dflt_labels, e_maps, e_maps_res_clp, example_scenarios):
    '''
    compiles a table with all the necessary catchment data:
    x, y, z, age, age uncertainty, fertility, erosional weights for all scenarios
    dflt_grids: list of clipped grids to export
    dflt_labels: list of labels for dflt_grids
    e_maps: dictionary of imported erosional maps
    e_maps_res_clp: dictionary of clipped erosional maps
    example_scenarios: flag to activate default example scenarios
    '''  
    # add labels and grids of the imported erosion maps and fertility
    if len(e_maps)>0:
        for key,item in e_maps_res_clp.items():
            dflt_labels.append(key)
            dflt_grids.append(item)

    ws_data = pd.DataFrame() # initiate dataframe
    Ncells = dflt_grids[0][dflt_grids[0]==dflt_grids[0]].size
    for g,l in zip(dflt_grids, dflt_labels):
        if g[g==g].size != Ncells: # check if nodata cells were involved
            warnings.warn('\nThe number of no-data cells in the '+l+' raster must match that of the clipped DEM\nPlease, make sure that the watershed polygon does not contain no-data.')         
        ws_data[l] = g[g==g] # drop the nans and reshape to 1D-array

    if len(e_maps)>0: # if present, recalculate erosional weights, such that the minimum possible erosion equals 1
        for k,i in e_maps_res_clp.items():
            ws_data[k] = ws_data[k]/ws_data[k].min()
            
    # then make uniform erosion scenario
    ws_data['Euni'] = np.ones(len(ws_data))
    
    # and sort the dataframe by elevation
    ws_data.sort_values(by='z', inplace=True)
    
    # If wanted, make default example erosional weights (exponential and inverse exponential function of elevation)
    if example_scenarios:
        ws_data['E_exp_Z'] = np.exp(ws_data.z/ws_data.z.min())
        ws_data.E_exp_Z = ws_data.E_exp_Z/ws_data.E_exp_Z.min()
        ws_data['E_inv_exp_Z'] = 1/ws_data.E_exp_Z
        ws_data.E_inv_exp_Z = ws_data.E_inv_exp_Z/ws_data.E_inv_exp_Z.min()
    
    return ws_data
    
############################################################################################################

def make_pops_dict(ws_data_filename, f_map_filename, multiplier):
    '''
    Makes a dictionary of predicted detrital grain populations, one for each scenario.
    ws_data_filename - .xlsx file of the exported pd.DataFrame of watershed data with following columns:
                x,y,z,age,age_u%, a column of erosional weight for each e_map, Euni, example scenarios
    f_map_filename - input filename of the fertility map (can be None)
    multi - multiplication factor to be used if no fertility map is present
    returns pops_dictionary and scenario_labels
    '''
    pops = OrderedDict() # preallocate dictionary of populations
    ws_data = pd.read_excel(ws_data_filename)
    scen_labels = list(ws_data.columns)[5:] # list of scenario labels to iterate through
    try:
        # use fertility values make grain populations, if present
        fkey = f_map_filename[:f_map_filename.find('.')] # remember key of fertility map
        scen_labels.remove(fkey) # remove it from the scenarios
        ws_data['Multiplier'] = ws_data[fkey]*multiplier
    except:
        # do not use fertility values to make populations
        ws_data['Multiplier'] = ws_data.Euni*multiplier
    for scen in scen_labels:
        # assign to each cell an amount of grains proportional to fertility and/or erosional weight
        # make column that to informs how many grains (N) per cell
        ws_data['N_'+scen] = np.rint(ws_data[scen]*ws_data.Multiplier)
        # make a gaussian distribution for each cell
        # Draw from it N grains and store them in a 1D-array, in the populations dictionary
        pops[scen] = np.array([])
        for A,U,N in zip(ws_data.age, ws_data['age_u%'], ws_data['N_'+scen]):
            pops[scen] = np.append(pops[scen], np.random.normal(A,np.abs(A*U/100),int(N)))
    return pops, scen_labels

############################################################################################################

def get_detr_pops(detrital_ages_filenames, pops, ipf):
    '''
    adds the detrital populations to the pops dictionary
    and returns also a list of detrital labels and a dictionary of the imported detrital data
    detrital_ages_filenames - list of detrital data filenames (input parameters), it can also be []
    pops - dictionary of arrays with grain populations, one for each scenario
    ipf - string, the path to the input folder
    '''
    dd, detr_labels = OrderedDict(), []
    for filename in detrital_ages_filenames:
        label = filename[:filename.find('.')]
        detr_labels.append(label)
        try:
            dd[label] = pd.read_excel(ipf+'/'+filename)
        except:
            dd[label] = pd.read_csv(ipf+'/'+filename)
        pops[label] = dd[label].age.values
    return pops, dd, detr_labels

############################################################################################################

def make_cdf(pop):
    '''
    returns sorted values and quantiles of a population in a pd.DataFrame
    pop - 1d-array
    '''
    df = pd.DataFrame()
    pop.sort()
    df['vals'] = pop
    df['cdf_y'] = (np.arange(pop.size)+1)/pop.size
    return df

############################################################################################################

def make_spdf(pop,sd=0.05):
    '''
    spdf (synoptic probability density function) equation
    from Ruhl & Hodges 2003, Vermeesch 2007, Riebe et al. 2015

    pop = 1d-array of grain ages
    sd = list/array of sigma1, or scalar informing preassigned 1sigma (e.g. 0.05 = 5% uncertainty)

    returns 1d-array where each element informs the frequency of the respective sorted value from
    '''
    pop = np.array(pop)
    if np.isscalar(sd):
        sigma1 = np.array([t*sd for t in pop])
    else:
        sigma1 = np.array(sd)
    vals = np.unique(pop, return_counts=False)
    spdf_y = [1.0/pop.size*np.sum(np.array([np.sqrt(2.0*np.pi)*u*np.exp((t-ti)**2.0/(2.0*u**2.0))
                                            for ti,u in zip(pop,sigma1)])**-1) for t in vals]
    return np.array(spdf_y)

############################################################################################################

def get_KS(res_p, d):
    '''
    res_p = resampled population p, with n = k
    d = a distribution with n != k
    returns KS statistic
    '''
    res_p = np.array(res_p)
    res_p.sort()
    # return KS statistic between distribution d and the distribution of res_p
    return np.max(np.abs((np.arange(res_p.size)+1)/res_p.size-np.interp(x=res_p,xp=d.vals,fp=d.cdf_y)))

############################################################################################################

def get_Kui(res_p, d):
    '''
    function to calculate the statistic of Kuiper's test,
    i.e., given a CDF and a number n of observations x_i-n, Kuiper's statistic expresses the sum of the two maxima
    of the vectors CDF-CDF(x_i-n) and CDF(x_i-n)-CDF
    that inform vertical distances both above and below the reference CDF

    res_p = resampled population p, with n=k
    d = real distribution of p

    returns Kuiper test's statistic (D)
    '''
    res_p = np.array(res_p)
    res_p.sort()
    Dplus = np.max((np.arange(res_p.size)+1)/res_p.size-np.interp(x=res_p,xp=d.vals,fp=d.cdf_y))
    Dmin = np.max(np.interp(x=res_p,xp=d.vals,fp=d.cdf_y)-np.arange(res_p.size)/res_p.size)
    return Dplus + Dmin

############################################################################################################

def smooth(y, window_size=3, order=1, deriv=0, rate=1):
    '''
    smoothing function, so that curves can be calculated on the base of ~1000 iterations,
    but still look good, if wanted
    '''
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        print("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2

    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y1 = np.concatenate((firstvals, y, lastvals))
    y_new = np.convolve( m[::-1], y1, mode='valid')
    y_new[0] = y[0]
    return y_new

############################################################################################################

def plot_distributions(pops_dict, dists_dict, ref_scen, detr_labels, saveas, noise_size=50, show_DKW=False, confidence=0.95, colmap=scm6.hawaii):
    '''
    plotting function to display all distributions
    pops_dict: dictionary of all simulated populations
    dists_dict: dictionary of distribution dataframes, having "vals" and "cdf_y" columns
    ref_scen: label of the reference scenario
    detr_labels: list of labels to recognize measured detrital destributions
    saveas: path to save to
    noise_size: sample size for which you want to generate noise around the reference scenario
    show_DKW: boolean, if True the DKW confidence interval is calculated and plotted as alternative to the iterations
    confidence: scalar, confidence interval for the DKW
    '''
    fig = plt.figure(figsize=(14,14))
    gspec = gs.GridSpec(2,1,figure=fig)
    ax1 = fig.add_subplot(gspec[0])

    # plot all scenarios' kernel density estimations
    color=iter(colmap(np.linspace(0,1,len(dists_dict))))
    xlim = (pops_dict[ref_scen].min(), pops_dict[ref_scen].max())
    for scen,df in dists_dict.items():
        if sum([i==scen for i in detr_labels])>0:
            ls='--'
        else:
            ls='-'
        sns.kdeplot(pops_dict[scen], color=next(color), linewidth=4, linestyle=ls, label='_nolegend_', ax=ax1)
        ax1.set(xlim=xlim, ylabel='frequency', xlabel='', xticks=[])
        ax1.set_title('Modeled vs Detrital: Kernel Density Estimation',
                     fontdict={'weight':'bold'})

    # second row
    ax2 = fig.add_subplot(gspec[1])
    color=iter(colmap(np.linspace(0,1,len(dists_dict))))
    c_ref = next(color)
    # plot all scenarios cdf
    if not show_DKW:
        for i in np.arange(100): # include 100 random subsamples of reference scenario?
            pop1 = np.random.choice(pops_dict[ref_scen],noise_size)
            if i==0: # save legend entry only for the first iteration
                dist1 = make_cdf(pop1)
                sns.lineplot(x=dist1.vals, y=dist1.cdf_y,
                             color=c_ref, alpha=0.1, lw=1, ax=ax2, label=ref_scen+', n='+str(noise_size))
            else: # plot the remaining iterations without showing them in the legend
                dist1 = make_cdf(pop1)
                sns.lineplot(x=dist1.vals, y=dist1.cdf_y,
                             color=c_ref, alpha=0.1, lw=1, ax=ax2, label='_nolegend_')

    color=iter(colmap(np.linspace(0,1,len(dists_dict))))
    for scen,df in dists_dict.items():
        if sum([i==scen for i in detr_labels])>0:
            ls = '--'
        else:
            ls = '-'
        if scen == ref_scen and show_DKW:
            DKW = np.sqrt(np.log(2/(1-confidence))/(2*noise_size))
            ax2.fill_between(x=df.vals, y1=df.cdf_y-DKW, y2=df.cdf_y+DKW, color=c_ref, alpha=0.3,
                             label=str(int(confidence*100))+'% conf of '+ref_scen+', n='+str(noise_size))
        ax2.plot(df.vals, df.cdf_y, color=next(color), linestyle=ls, label=scen+', n='+str(len(df)), lw=4)
    ax2.set(xlim=xlim, ylabel='cumulative frequency', xlabel='age [My]')
    ax2.set_title('Modeled vs Detrital: Cumulative Age Distribution',
                     fontdict={'weight':'bold'})
    ax2.legend()
    fig.savefig(saveas, dpi=200) # save fig

############################################################################################################

def get_probabilities(pops_dict, dists_dict, all_k, k_iter, scen_labels, ref_scen):
    '''
    makes a dictionary containing the probabilities that each scenario can be discerned from ref_scen,
    based on the KS statistics, for different sample sizes.
    pops_dict: dictionary of all simulated populations
    dists_dict: dictionary of distribution dataframes, having "vals" and "cdf_y" columns
    all_k: 1D-array of k values (integers)
    k_iter: number of iterations
    scen_labels: list of scenario labels to be compared
    ref_scen: the label of the reference scenario, based on which prob_dict has been made
    '''
    # skip this if there is only one scenario
    if len(scen_labels)>1:
        # calculate 95% confidence divergence for each k, between reference n=∞ and reference n=size of dd
        Dc_within, D_dist_dict, Dc = {}, {}, [] # prepare dictionaries and list to store results
        for k in all_k: # get array of random divergencies, using KS statistic
            D_arr = np.array([get_KS(np.random.choice(pops_dict[ref_scen],k), dists_dict[ref_scen]) for element in np.arange(k_iter)])
            D_arr.sort() # sort values
            D_arr_prob = (np.arange(D_arr.size)+1)/D_arr.size # calculate cumulative probability of each value
            ind95 = np.where(D_arr_prob<=0.95)[0][-1] # index of the 95th percentile
            Dc.append(D_arr[ind95]) # store values
            D_dist_dict[ref_scen+'_'+str(k)] = [D_arr, D_arr_prob]
        Dc_within[ref_scen] = Dc # list of critical distances, each for a k

        scen_labels1 = scen_labels.copy()
        scen_labels1.pop(scen_labels1.index(ref_scen)) # now remove ref_scen from labels for the next plot
        for scen in scen_labels1:
            for k in all_k: # get array of random divergences, using KS statistic
                D_arr = np.array([get_KS(np.random.choice(pops_dict[scen],k), dists_dict[ref_scen]) for i in np.arange(k_iter)])
                D_arr.sort() # sort values
                D_arr_prob = (np.arange(D_arr.size)+1)/D_arr.size # calculate cumulative probability of each value
                D_dist_dict[scen+'_'+str(k)] = [D_arr, D_arr_prob]

        # Calculate confidence as function of the number of grains
        # Compare Dc_within to scenarios for all k and get probability of scenarios being discerned from ref_scen
        probs = {}
        for scen in scen_labels1:
            probs1 = []
            for i in np.arange(len(all_k)): # for all k values
                # get index of element where distance scenario-ref_scen is greater than critical distance Dc_within
                ind = np.where(D_dist_dict[scen+'_'+str(all_k[i])][0]>Dc_within[ref_scen][i])[0][0]
                # get complementary probability at position ind (i.e. how many iterations are greater than that)
                probs1.append(100*(1-D_dist_dict[scen+'_'+str(all_k[i])][1][ind]))
            probs[scen] = probs1
        return probs
    else:
        return None

############################################################################################################

def plot_confidence(prob_dict, all_k, ref_scen, saveas, num_of_colors, colmap=scm6.hawaii):
    '''
    plots the confidence level as function of sample size
    prob_dict: dictionary of arrays, each containing confidence values related to all_k
    all_k: 1D-array of k values (integers)
    ref_scen: the label of the reference scenario, based on which prob_dict has been made
    saveas: path to be saved at
    num_of_colorss: number of colors to iterate through,
                    must equal the number of colors plotted before, to be consistent
    '''

    if prob_dict==None:
        warnings.warn('There are no erosion scenarios to compare to {}, so the plot would be empty'.format(ref_scen))
    else:
        sns.set_style('ticks')
        fig,ax = plt.subplots(figsize=(12,8))

        # plot grey fields
        ax.fill_between([all_k[0],all_k[-1]],[95,95],[68,68],color='k',alpha=0.1)
        ax.text(all_k[0]+0.5,96,'95%',fontdict=dict(size=20))
        ax.fill_between([all_k[0],all_k[-1]],[68,68],color='k',alpha=0.2)
        ax.text(all_k[0]+0.5,69,'68%',fontdict=dict(size=20))
        color=iter(colmap(np.linspace(0,1,num_of_colors)))
        next(color) # skip one color for Euni
        leg=[]
        for key,i in prob_dict.items():
            c = next(color)
            ax.plot(all_k, smooth(i), c=c, alpha=1)
            leg.append(key)
        ax.set(xlim=(all_k[0],all_k[-1]), ylim=(0,101), yticks=[0,20,40,60,80,100])
        ax.set_xlabel('number of grains',fontdict=dict(size=20, weight='bold'))
        ax.set_ylabel('confidence level [%]',fontdict=dict(size=20, weight='bold'))
        ax.set_title('Confidence of discerning from "'+ref_scen+'" as function of sample size',
                     fontdict=dict(size=20,weight='bold'), pad=10)

        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(5))

        ax.legend(leg, loc='lower right')
        sns.set_style('white')
        fig.savefig(saveas, dpi=200)

############################################################################################################

def make_pops_1sigma(dd, n):
    '''
    makes dictionary of detrital populations that account for 1 sigma error
    '''
    pops_1s = OrderedDict() # allocate dictionary to store all 1sigma populations
    for key,item in dd.items():
        # make empty array to store new population
        pop_1s = np.array([])
        for j in item.index:
            # for each detrital age, draw n normally distributed ages based on analytical error and store them
            pop_1s = np.append(pop_1s,np.random.normal(item.loc[j].age, item.loc[j].age_u, n))
        # store population in dictionary
        pops_1s[key] = pop_1s
    return pops_1s

############################################################################################################

def get_scen2detr_diss(dd, pops, pops_1s, dists, scen_labels, n):
    '''
    makes the dataframe to be plotted as violins below.
    This is a collection of all dissimilarities calculated between subsampled scenarios
    and the observed detrital populations
    '''
    scen2detr = OrderedDict() # allocate dictionary to store all dissimilarities
    for key,item in dd.items():
        # make empty lists to store values of each iteration
        KS_list, Kui_list, scenario_list = [], [], []
        # iterate through erosion scenarios
        for scen in scen_labels:
            KS_list = KS_list + [get_KS(np.random.choice(pops[scen],len(item)),dists[key]) for i in np.arange(n)]
            Kui_list = Kui_list + [get_Kui(np.random.choice(pops[scen],len(item)),dists[key]) for i in np.arange(n)]
            scenario_list = scenario_list + [scen for i in np.arange(n)]

        # and do the same analysis for the new pop_1s
        KS_list = KS_list + [get_KS(np.random.choice(pops_1s[key],len(item)),dists[key]) for i in np.arange(n)]
        Kui_list = Kui_list + [get_Kui(np.random.choice(pops_1s[key],len(item)),dists[key]) for i in np.arange(n)]
        scenario_list = scenario_list + [key for i in np.arange(n)]

        # allocate dataframe to store divergences scenario-to-detrital from all iterations
        scen2detr[key] = pd.DataFrame(columns=['divergence','metric','scenario'])
        # write dataframe
        scen2detr[key].divergence = KS_list + Kui_list
        scen2detr[key].metric = ['KS stat' for i in KS_list]+['Kuiper stat' for i in Kui_list]
        scen2detr[key].scenario = scenario_list + scenario_list
    return scen2detr

############################################################################################################

def get_quantiles_and_overlaps(dd, scen2detr, pops, dists, scen_labels, n):
    '''
    makes dictionary of 68% and 95% quantiles of the distribution of KS stats to each observed detrital distribution
    also, for each observed detrital, it makes a list of overlapping percentages.
    These inform how many of the iterations fall in the 95% range of noise due to sample size
    '''
    q68q95 = OrderedDict() # allocate dictionary
    for scen in scen_labels: # iterate through scenarios
        # get the 68% and 95% percentiles for the scenarios
        KSarr_s = np.array([get_KS(np.random.choice(pops[scen],100),dists[scen]) for i in np.arange(n)])
        KSarr_s.sort()
        q68q95[scen] = [KSarr_s[np.nonzero((np.arange(n)+1)/n >= q)[0][0]] for q in (0.68,0.95)]

    if len(dd)<1:
        return q68q95, {}

    else:
        # dictionary to store degree of overlap among calculated KS distributions
        overlaps = OrderedDict()
        for key,item in dd.items(): # iterate through imported detrital populations
            overlaps[key] = [] # prepare empty list
            KSarrays = scen2detr[key].where(scen2detr[key].metric=='KS stat').dropna() # do not consider Kuiper stat
            # select array of KS stats to current detrital distribution
            KSarr_d = KSarrays.where(KSarrays.scenario==key).dropna().divergence.values
            KSarr_d.sort() # sort array
            # get the 68% and 95% percentiles
            q0 = KSarr_d.min()
            q68q95[key] = [KSarr_d[np.nonzero((np.arange(n)+1)/n >= q)[0][0]] for q in (0.68,0.95)]
            # iterate through scenarios
            for scen in scen_labels:
                # select array of dissimilarity (KS stat) to the detrital distribution
                KSarr_2d = KSarrays.where(KSarrays.scenario==scen).dropna().divergence.values
                KSarr_2d.sort() # sort it
                # calculate how many elements are comprised in the range q0 - q95 of the detrital noise
                overlaps[key].append(np.where((KSarr_2d>=q0)&(KSarr_2d<=q68q95[key][-1]),1,0).sum()/n)
        return q68q95, overlaps

############################################################################################################

def plot_violins(data, label, column, saveas, k_iter, sam_size, overlaps, colmap='colorblind'):
    '''
    plotting function for violins figure
    data: pd.dataframe with column 'scenario' (for the x axis) and another column to be plotted as y
    label: string, the key of the wanted population for the dictionary of detrital data
    column: string, column label of the dataframe to plot
    saveas: path to save to
    k_iter: number of iterations
    sam_size: number of grains used in each iteration
    overlaps: dictionary of overlapping fraction, to judge degree of fit
    '''
    # split data
    data1, data2 = data.where(data.scenario!=label).dropna(), data.where(data.scenario==label).dropna()
    # prepare figure
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15,8))
    gspec = gs.GridSpec(1,4,figure=fig)

    # left panel
    ax1 = fig.add_subplot(gspec[:3])
    sns.violinplot(data=data1, x='scenario', y=column, split=True, hue='metric', ax=ax1, cut=0, scale='area', palette=colmap, inner=None)
    for i in np.arange(len(data1.scenario.unique())):
        # print degree of overlap between predicted violin and observed violin
        ax1.text(i-0.1,0.02,str(np.around(100*overlaps[label][i],1))+'%', fontdict={'weight':'bold', 'style':'italic'})
    ax1.legend([], frameon=False)
    ax1.set_ylim(0,data.divergence.max())
    ax1.set_xlabel('predicted erosion scenario', fontdict={'weight':'bold'})
    ax1.set_ylabel('dissimilarity from '+str(k_iter)+' iterations',fontdict={'weight':'bold'})
    ax1.set_title('Dissimilarity of subsampled erosion scenarios to '+label+', n='+str(sam_size),pad=10, fontdict={'weight':'bold'})

    # right panel
    ax2 = fig.add_subplot(gspec[3])
    sns.violinplot(data=data2, x='scenario', y=column, split=True, hue='metric', ax=ax2, cut=0, scale='area', palette=colmap, inner=None)
    ax2.set_xlabel('observed distribution', fontdict={'weight':'bold'})
    ax2.set(ylim=(0,data.divergence.max()), ylabel='', yticklabels=[])
    ax2.set_title('Noise due to n='+str(sam_size), pad=10, fontdict={'weight':'bold'})
    sns.set_style('white') # set back the seaborn style back to white with no lines
    fig.savefig(saveas, dpi=200) # save figure

############################################################################################################

def get_diss_matrix(pops_dict):
    '''
    prepares dissimilarity matrix based on KS statistics,
    to be used for MDS
    '''
    l = list(pops_dict.keys())
    diss = np.zeros((len(l),len(l)))
    for i in np.arange(len(l)):
        for j in np.arange(len(l)):
            diss[i,j] = get_KS(pops_dict[l[i]], make_cdf(pops_dict[l[j]]))
    for i in np.arange(len(l)):
        for j in np.arange(len(l)):
            diss[j,i] = diss[i,j] # only half of the matrix counts
    return diss

############################################################################################################

def plot_MDS_results(x, colmap, pops, q68q95, scen_labels, saveas):
    '''
    makes a plot of the MDS prediction, where input dissimilarities are embedded in a 2D-space.
    A goodness of fit "Shepard Plot" is also shown to evaluate the model fit.

    x - fitted MDS model
    colmap - an iterable colormap
    pops - dictionary of grain populations, including both scenario predictions and observations
    q68q95 - dictionary of lists (one per population). Each list contains the same number of items,
                each expressing the 68% and 95% percentiles of dissimilarities obtained subsampling
                the population to n=num. of observations and calculating the KS stat to n=∞
    scen_labels - list of scenario prediction labels, excluding observations
    title - string, title of the plot
    '''
    fig = plt.figure(figsize=(14,8))
    gspec = gs.GridSpec(1,5,figure=fig)

    l, emb, diss = list(pops.keys()), x.embedding_, x.dissimilarity_matrix_

    # left panel
    ax1 = fig.add_subplot(gspec[:4])
    circles = {}
    color = iter(colmap(np.linspace(0,1,len(pops.keys()))))
    count = 0
    for key,item in pops.items():
        col=next(color)
        for q in q68q95[key]: # iterate circle plots
            count+=1
            if count <= (l.index(scen_labels[-1])+1)*2: # only fill circles of observed detrital distr
                circles[count] = plt.Circle((emb[l.index(key),0], emb[l.index(key),1]), q, fill=False, ec=col, alpha=0.5, lw=2)
            else:
                circles[count] = plt.Circle((emb[l.index(key),0], emb[l.index(key),1]), q, color=col, alpha=0.3, lw=2)
            ax1.add_patch(circles[count])

    ax1.scatter(emb[:,0],emb[:,1], c=np.arange(len(l)), cmap=colmap) # scatter plot
    for X,Y,label in zip(emb[:,0],emb[:,1],l):
        ax1.annotate(label,(X,Y))
    ax1.set_title('MDS plot', pad=10, fontdict={'weight':'bold'})
    ax1.set_xlabel('MDS coordinate 1', fontdict={'weight':'bold'})
    ax1.set_ylabel('MDS coordinate 2', fontdict={'weight':'bold'})

    # right panel
    ax2 = fig.add_subplot(gspec[4])
    # calculate distance between points based on MDS results
    diss_emb = diss.copy()
    for i in np.arange(len(l)):
        for j in np.arange(len(l)):
            diss_emb[i,j] = np.sqrt((emb[i,0]-emb[j,0])**2+(emb[i,1]-emb[j,1])**2)

    # plot input dissimilarities against modeled ones, to see how good is the fit
    ax2.plot(diss.reshape(diss.size), diss.reshape(diss.size),'k')
    ax2.scatter(diss.reshape(diss.size), diss_emb.reshape(diss.size), alpha=0.5)
    ax2.set_title('Goodness of fit', pad=10, fontdict={'weight':'bold'})
    ax2.set_xlabel('input dissimilarity', fontdict={'weight':'bold'})
    ax2.set_ylabel('MDS-modeled dissimilarity', fontdict={'weight':'bold'})
    plt.tight_layout()
    fig.savefig(saveas, dpi=200) # save figure
