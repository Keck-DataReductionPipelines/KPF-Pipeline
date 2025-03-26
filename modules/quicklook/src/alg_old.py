###### Below is the old QLP code, partially deconstructed.                #######
###### There are some notes about elements to put in to analysis modules. #######

            #looking at the fixed noise patterns
            '''
            if version =='Bias':
                a = np.copy(counts)
                a_med = np.nanmedian(a.ravel())
                a_std = np.nanstd(a.ravel())
                pdf, bin_edges = np.histogram(a.ravel(),bins=80, range = (a_med-8*a_std,a_med+8*a_std))
                print('bin edges',bin_edges)
                count_fit = (bin_edges[1:]+bin_edges[:-1])/2
                from astropy import modeling
                fitter = modeling.fitting.LevMarLSQFitter()#the gaussian fit of the ccf
                model = modeling.models.Gaussian1D()
                fitted_model = fitter(model,count_fit-a_med, pdf)
                amp =fitted_model.amplitude.value
                gamma =fitted_model.mean.value+a_med
                std =fitted_model.stddev.value
                print('1st gaussian',amp,gamma,std)
                plt.close('all')
                plt.plot(count_fit,amp*np.exp(-0.5*(count_fit-gamma)**2/std**2),':',color = 'red', label = '1st component')#1/std/np.sqrt(2*np.pi)*
                #plt.ylim(1,10*amp)
                fitter = modeling.fitting.LevMarLSQFitter()#the gaussian fit of the ccf
                model = modeling.models.Gaussian1D()
                fitted_model = fitter(model,count_fit[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)]-a_med, pdf[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)])
                amp2 =fitted_model.amplitude.value
                gamma2 =fitted_model.mean.value+a_med
                std2 =fitted_model.stddev.value
                print('2nd gaussian',amp2,gamma2,std2)

                plt.plot(count_fit,amp2*np.exp(-0.5*(count_fit-gamma2)**2/std2**2),':',color = 'green', label = '2nd component')#1/std/np.sqrt(2*np.pi)*
                #plt.ylim(1,10**7)
                plt.plot((bin_edges[1:]+bin_edges[:-1])/2,pdf, label = 'All')
                plt.scatter(np.array((bin_edges[1:]+bin_edges[:-1])/2,'d')[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)],pdf[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)], label = 'Larger Var Component')
                plt.legend()
                plt.xlabel('Counts (e-)')
                plt.ylabel('Number of Pixels')
                plt.yscale('log')
                plt.savefig(output_dir+'fig/'+exposure_name+'_bias_'+ccd_color[i_color]+'.png')
                plt.close('all')
            '''


# commented out on July 16 -- determine how to add to QLP
#                plt.figure(figsize=(5,4))
#                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#                threshold = 2
#                high_var_counts = np.copy(counts)
#                low_var_counts = np.copy(counts)
#                high_var_counts[abs(counts-np.nanmedian(counts))<threshold*np.nanstd(counts)] = np.nan
#                low_var_counts[abs(counts-np.nanmedian(counts))>threshold*np.nanstd(counts)] = np.nan
#
#                plt.imshow(high_var_counts, vmin = np.percentile(flatten_counts,0.1),vmax = np.percentile(flatten_counts,99.9),interpolation = 'None',origin = 'lower',cmap = 'bwr')
#                plt.xlabel('x (pixel number)')
#                plt.ylabel('y (pixel number)')
#                plt.title(ccd_color[i_color]+' '+version+' High Variance '+exposure_name)
#                plt.colorbar(label = 'Counts (e-)')
#
#                plt.text(2200,3600, 'Nominal STD: %5.1f' % np.nanstd(np.ravel(low_var_counts)))
#                plt.text(2200,3300, 'Fixed Pattern STD: %5.1f' % np.nanstd(np.ravel(high_var_counts)))
#                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_high_var_'+ccd_color[i_color]+'_zoomable.png')
#                plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_2D_Frame_high_var_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
#                plt.close()
#                '''
#                plt.figure(figsize=(5,4))
#                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#
#                plt.imshow(low_var_counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
#                plt.xlabel('x (pixel number)')
#                plt.ylabel('y (pixel number)')
#                plt.title(ccd_color[i_color]+' '+version+' Low Variance')
#                plt.colorbar(label = 'Counts (e-)')
#                plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_low_var_'+ccd_color[i_color]+'_zoomable.png')
#                '''
#            #print('master file',version,i_color,master_file,len(master_flatten_counts))
#            if master_file != 'None' and len(master_flatten_counts)>1:
#                plt.figure(figsize=(5,4))
#                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#                #pcrint(counts,master_counts)
#                counts_norm = np.percentile(counts,99)
#                master_counts_norm = np.percentile(master_counts,99)
#                if np.shape(counts)!=np.shape(master_counts): continue
#                difference = counts/counts_norm-master_counts/master_counts_norm
#
#                plt.imshow(difference, vmin = np.percentile(difference,1),vmax = np.percentile(difference,99), interpolation = 'None',origin = 'lower')
#                plt.xlabel('x (pixel number)')
#                plt.ylabel('y (pixel number)')
#                plt.title(ccd_color[i_color]+' '+version+'- Master '+version+' '+exposure_name)
#                plt.colorbar(label = 'Fractional Difference')
#                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'_zoomable.png')
#                plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
#             #Hisogram
#            plt.close()
#            plt.figure(figsize=(5,4))
#            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#
#            #print(np.percentile(flatten_counts,99.9),saturation_limit)
#            plt.hist(flatten_counts, bins = 50,alpha =0.5, label = 'Median: ' + '%4.1f; ' % np.nanmedian(flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(flatten_counts)+'; Saturated? '+str(np.percentile(flatten_counts,99.99)>saturation_limit),density = False, range = (np.percentile(flatten_counts,0.005),np.percentile(flatten_counts,99.995)))#[flatten_counts<np.percentile(flatten_counts,99.9)]
#            if master_file != 'None' and len(master_flatten_counts)>1: plt.hist(master_flatten_counts, bins = 50,alpha =0.5, label = 'Master Median: '+ '%4.1f' % np.nanmedian(master_flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(master_flatten_counts), histtype='step',density = False, color = 'orange', linewidth = 1 , range = (np.percentile(master_flatten_counts,0.005),np.percentile(master_flatten_counts,99.995))) #[master_flatten_counts<np.percentile(master_flatten_counts,99.9)]
#            #plt.text(0.1,0.2,np.nanmedian(flatten_counts))
#            plt.xlabel('Counts (e-)')
#            plt.ylabel('Number of Pixels')
#            plt.yscale('log')
#            plt.title(ccd_color[i_color]+' '+version+' Histogram '+exposure_name)
#            plt.legend(loc='lower right')
#            #plt.savefig(output_dir+'fig/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png')
#            plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png', dpi=200)
#
#            #Column cut
#            plt.close()
#            plt.figure(figsize=(8,4))
#            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)
#
#            column_sum = np.nansum(counts,axis = 0)
#            #print('which_column',np.where(column_sum==np.nanmax(column_sum))[0][0])
#            which_column = np.where(column_sum==np.nanmax(column_sum))[0][0] #int(np.shape(master_counts)[1]/2)
#
#            plt.plot(np.ones_like(counts[:,which_column])*saturation_limit,':',alpha = 0.5,linewidth =  1., label = 'Saturation Limit: '+str(saturation_limit), color = 'gray')
#            plt.plot(counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = ccd_color[i_color]+' '+version, color = 'Blue')
#            if master_file != 'None' and len(master_flatten_counts)>1: plt.plot(master_counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = 'Master', color = 'Orange')
#            plt.yscale('log')
#            plt.ylabel('log(Counts/e-)')
#            plt.xlabel('Row Number')
#            plt.title(ccd_color[i_color]+' '+version+' Column Cut Through Column '+str(which_column) + ' '+exposure_name)#(Middle of CCD)
#            plt.ylim(1,1.2*np.nanmax(counts[:,which_column]))
#            plt.legend()
#            '''
#            #show the order order_trace
#            if version == 'Flat_All':
#                 for i in range(np.shape(order_trace)[0]):#[50]:#range(np.shape(order_trace)[0])
#                     #print(order_trace.iloc[i]['X1'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
#                     x_grid = np.linspace(0,order_trace.iloc[i]['X2'],int(order_trace.iloc[i]['X2']-0))
#                     y_grid = order_trace.iloc[i]['Coeff0']+x_grid*order_trace.iloc[i]['Coeff1']+x_grid**2*order_trace.iloc[i]['Coeff2']+x_grid**3*order_trace.iloc[i]['Coeff3']
#                     #print(i,len(y_grid),which_column,y_grid[which_column])
#                     #plt.plot([y_grid[which_column],y_grid[which_column]],[1,1.*np.nanmax(counts[:,which_column])],color ='red',linewidth = 0.2)
#                     plt.plot([y_grid[which_column]-order_trace.iloc[i]['BottomEdge'],y_grid[which_column]-order_trace.iloc[i]['BottomEdge']],[1,1.*np.nanmax(counts[:,which_column])],color ='red',linewidth = 0.2)
#                     plt.plot([y_grid[which_column]+order_trace.iloc[i]['TopEdge'],y_grid[which_column]+order_trace.iloc[i]['TopEdge']],[1,1.*np.nanmax(counts[:,which_column])],color ='magenta',linewidth = 0.2)
#                     #plt.plot(x_grid[which_column],y_grid[which_column]-order_trace.iloc[i]['BottomEdge'],color ='white',linewidth = 0.2,alpha = 1)
#                     #plt.plot(x_grid[which_column],y_grid[which_column]+order_trace.iloc[i]['TopEdge'],color ='black',linewidth = 0.2,alpha = 1)
#            '''
#            #plt.savefig(output_dir+'fig/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'.png')
#            plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'_zoomable.png', dpi=200)
#            plt.close()


        

