#! MODIFIED Functions for Coral Adaptation model
import numpy as np
import random

#! GROWTH FUNCTION
#! Calculate growth rate as a function of species trait (optimum temperature) and local 
#! environmental condition (reef temperature).
def growth_fun(r_max,T,z,w,species_type):
    if z.shape[0] > 1: # If there is more than one reef
        T = np.repeat(T,z.shape[1]).reshape(z.shape[0],z.shape[1])
    else: # If there is a single reef
        T = np.array([np.repeat(T,z.shape[1])])
        
    r = np.zeros((z.shape[0],z.shape[1])) #Preallocate growth vector
    coral_col = np.where(species_type == 1)[1]
    algae_col = np.where(species_type == 2)[1]
    r[:,coral_col] =( (r_max[:,coral_col]/np.sqrt(2.*np.pi*pow(w[:,coral_col],2.)))
                    *np.exp((-pow((T[:,coral_col]-z[:,coral_col]),2.))/(2*pow(w[:,coral_col],2.))) )
    r[:,algae_col] = 0.49 * r_max[:,algae_col]

    return r

#! MORTALITY FUNCTION    
def mortality_fun(r_max,T,z,w,species_type,mpa_status,alg_mort):
    m = np.zeros((z.shape[0],z.shape[1])) # Preallocate mortality vector
    algae_col = np.array([np.where(species_type == 2)[1]]) # Find algae columns
    
    if z.shape[0] > 1: # If there is more than one reef
        # Reshape T array to correspond with z matrix
        T = np.repeat(T,z.shape[1]).reshape(z.shape[0],z.shape[1]) 
        m[z<T] = 1 - np.exp(-pow((T-z),2)/pow(w,2))[z<T]
        m[z>=T] = 0
        
        # Indices of mpa reefs (corresponds to rows in N_all)
        is_mpa = np.array([np.where(mpa_status == 1)[1]]) 
        # Indices of non-mpa reefs (corresponds to rows in N_all)
        not_mpa = np.array([np.where(mpa_status != 1)[1]])
        
        # Create arrays of indices that correspond to is_mpa & algae_col and not_mpa & algae_col
        is_mpa_rows = np.array([is_mpa.repeat(algae_col.shape[1])]) 
        not_mpa_rows = np.array([not_mpa.repeat(algae_col.shape[1])])
        algae_col_is_mpa = np.tile(algae_col,is_mpa.shape[1])
        algae_col_not_mpa = np.tile(algae_col,not_mpa.shape[1])

        # Macroalgae calculations for multiple reefs
        m[is_mpa_rows,algae_col_is_mpa] = 0.3
        m[not_mpa_rows,algae_col_not_mpa] = alg_mort[not_mpa_rows]
        
    else: # If there is a single reef
        T = np.array([np.repeat(T,z.shape[1])])
    
        #Coral calculations
        m[z<T] = 1 - np.exp(-pow((T-z),2)/pow(w,2))[z<T]
        m[z>=T] = 0
        
        # Macroalgae calculations for a single reef
        if mpa_status == 1:
             m[0,algae_col] = 0.3
        else:
            m[0,algae_col] = alg_mort

    # Apply a correction such that the minimum amount of mortality experienced is 0.03        
    m[m<0.03] = 0.03
    
    return m
    
#! FITNESS FUNCTION     
def fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort):
    r = growth_fun(r_max,T,z,w,species_type)
    
    # If mortality varies with temperature
    if mortality_model == "temp_vary":
        m = mortality_fun(r_max,T,z,w,species_type,mpa_status,alg_mort)    
    # If mortality is constant
    else: 
        m = m_const
    
    #If there is more than one reef
    if N_all.shape[0] > 1:
        sum_interactions = np.array([np.sum(N_all[index,:] * alphas, axis=1) for index in range(N_all.shape[0])])
    else:
        sum_interactions = np.sum(alphas * N_all, axis=1)

    g = r * (1-sum_interactions) - m
    
    return g
    
#! dGdZ FUNCTION     
def dGdZ_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort):
    h = 1e-5
    dGdZ = np.zeros(z.shape)
    
    #If there is more than one reef
    if N_all.shape[0] > 1:
        # For each reef
        for i in np.arange(z.shape[0]): 
            # For each species
            for j in np.arange(z.shape[1]):
                h_matrix = np.zeros(z.shape)
                h_matrix[i,j] = h
                # Take the symmetric difference quotient at point z[i,j]
                term1 = fitness_fun(r_max,T,z+h_matrix,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
                term2 = fitness_fun(r_max,T,z-h_matrix,w,alphas,species_type,mpa_status,
                                         N_all,m_const,mortality_model,alg_mort)
                delta = (term1-term2)/(2*h)
                dGdZ[i,j] = delta[i,j]
    else:
        for j in np.arange(z.shape[1]):
            h_array = np.zeros(z.shape)
            h_array[0,j] = h
            # Take the symmetric difference quotient at point z[i,j]
            term1 = fitness_fun(r_max,T,z+h_array,w,alphas,species_type,mpa_status,
                                 N_all,m_const,mortality_model,alg_mort)
            term2 = fitness_fun(r_max,T,z-h_array,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
            delta = (term1-term2)/(2*h)
            dGdZ[0,j] = delta[0,j]
            
    return dGdZ

#! dGdZ2 FUNCTION 
def dGdZ2_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort):
    h = 1e-5
    dGdZ2 = np.zeros(z.shape)
    
    #If there is more than one reef
    if N_all.shape[0] > 1:
        # For each reef
        for i in np.arange(z.shape[0]): 
            # For each species
            for j in np.arange(z.shape[1]):
                h_matrix = np.zeros(z.shape)
                h_matrix[i,j] = h
                # Take the symmetric difference quotient at point z[i,j]
                term1 = fitness_fun(r_max,T,z+h_matrix,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
                term2 = fitness_fun(r_max,T,z-h_matrix,w,alphas,species_type,mpa_status,
                                         N_all,m_const,mortality_model,alg_mort)
                term3 = fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,
                                         N_all,m_const,mortality_model,alg_mort)
                delta = (term1+term2-2*term3)/pow(h,2)
                dGdZ2[i,j] = delta[i,j]
                
    else:
        for j in np.arange(z.shape[1]):
            h_array = np.zeros(z.shape)
            h_array[0,j] = h
            # Take the symmetric difference quotient at point z[i,j]
            term1 = fitness_fun(r_max,T,z+h_array,w,alphas,species_type,mpa_status,
                                 N_all,m_const,mortality_model,alg_mort)
            term2 = fitness_fun(r_max,T,z-h_array,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
            term3 = fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,
                                     N_all,m_const,mortality_model,alg_mort)
            delta = (term1+term2-2*term3)/pow(h,2)
            dGdZ2[0,j] = delta[0,j]
            
    return dGdZ2
    
#! dNdt FUNCTION     
def dNdt_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort,V,D,beta,areas):    
    
    if N_all.shape[0] > 1:
        V = np.tile(V, N_all.shape[0]).reshape(N_all.shape[0],N_all.shape[1])
    
    areas_reshaped = np.repeat(areas,N_all.shape[1]).reshape(areas.shape[0],N_all.shape[1])
    
    g = fitness_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort)
    dGdZ2 = dGdZ2_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort)      
    popdy = np.multiply(N_all,g)
    genload = 0.5 * V * dGdZ2 * N_all
    dispersal = (beta/areas_reshaped) * np.dot(D,(N_all*areas_reshaped))
    free_space = 1 - N_all.sum(axis=1)
    ID = np.where(free_space < 0)
    free_space[ID] = 0.
    larval_input = np.array([dispersal[index,:] * x for index, x in enumerate(free_space)])
    
    algae_ID = np.where(species_type==2)[1] #find algae columns
    larval_input[:,algae_ID] = 0
    
    dNdt = popdy + genload + larval_input
    
    #! Prevent NaN or population values below 1e-6 in output
    if np.isnan(dNdt).any():
        ID = np.where(np.isnan(dNdt))
        dNdt[ID] = 1e-6
    if (dNdt+N_all < 1e-6).any():
        ID = np.where(dNdt+N_all < 1e-6)
        dNdt[ID] = 1e-6

    return dNdt
    
#! Q FUNCTION     
def q_fun(N_all, N_min=1e-6):
    q = np.maximum(0, 1- N_min/(np.maximum(N_min,2*N_all)))
    return q
    
    
#! dZdt FUNCTION 
def dZdt_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort,V,D,beta,areas):
        
    if N_all.shape[0] > 1:
        V = np.tile(V, N_all.shape[0]).reshape(N_all.shape[0],N_all.shape[1])
    
    areas_reshaped = np.repeat(areas,N_all.shape[1]).reshape(areas.shape[0],N_all.shape[1])

    q = q_fun(N_all)
    dGdZ = dGdZ_fun(r_max,T,z,w,alphas,species_type,mpa_status,N_all,m_const,mortality_model,alg_mort)
    directional_selection = q * V * dGdZ
    gene_flow_term1 = (np.dot(D, N_all*areas_reshaped*z) / np.dot(D, N_all*areas_reshaped)) - z
    gene_flow_term2 = (beta/areas_reshaped * np.dot(D, N_all* areas_reshaped)) / (beta/areas_reshaped * np.dot(D, N_all* areas_reshaped) + N_all)
    free_space = 1. - N_all.sum(axis=1)
    ID = np.where(free_space < 0)
    free_space[ID] = 0.
    gene_flow = np.array([(gene_flow_term1*gene_flow_term2)[index,:] * x for index, x in enumerate(free_space)])
    algae_ID = np.where(species_type==2)[1] #find algae columns
    gene_flow[:,algae_ID] = 0
    
    dZdt = directional_selection + gene_flow
    return dZdt
    
#! GENERATE TEMPS FUNCTION     
def generate_temps_fun(size,mid=25,temp_range=2.5,temp_scenario='linear'):
    if temp_scenario == 'uniform':
        temps = np.repeat(mid,size)
    if temp_scenario == 'linear':
        temps = np.linspace(mid-temp_range, mid+temp_range, size)
    elif temp_scenario == 'random':
        a = np.linspace(mid-temp_range, mid+temp_range, size)
        temps = np.random.choice(a, size, replace=False, p=None)
    return temps

#! GENERATE INITIAL TRAITS FUNCTION         
def generate_traits_fun(nsp,size,temps,mid=25,temp_range=2.5,trait_scenario='perfect_adapt'):
    if trait_scenario == 'u_constant':
        a = np.linspace(mid-(temp_range/4), mid+(temp_range/4), nsp+2)[1:-nsp+2]
        b = np.repeat(a,size)
        traits = np.reshape(b,(nsp,size))
    if trait_scenario == 'same_constant':
        traits = np.full((nsp,size),mid)
    elif trait_scenario == 'perfect_adapt':
        a = np.tile(temps,nsp)
        traits = np.reshape(a,(nsp,size))
    return traits.T
    
#! GENERATE INITIAL STATE FUNCTION     
def generate_state_fun(size, nsp, cover=0.01,random=False):
    state = np.full((size,nsp),cover)
    if random:
        state = np.full(nsp*size,np.random.uniform(1e-6,.33,nsp*size)).reshape(size,nsp)
    return state
    
#! SET MPA FUNCTION        
def set_MPA_fun(mpa_param,amount=0.2,strategy='random',subregion_only=False):
    
    temps = mpa_param['temps']
    N_all = mpa_param['N_all']
    species_type = mpa_param['species_type']
    size = mpa_param['size']
    D = mpa_param['D']
    subregion_ID = mpa_param['subregion_ID']
    egvec_centrality_sorted = mpa_param['egvec_centrality_sorted']
    bw_centrality_sorted = mpa_param['bw_centrality_sorted']
    even_space = mpa_param['even_space']
    
    mpa = np.zeros(size)
    ID = np.where(species_type==1)[1]
    corals = N_all[:,ID].sum(axis=1) # Get the subset of N_all that corresponds to coral cover per reef
    #The following line applies to the 'portfolio' strategy that I haven't coded here
    ncoral = np.asarray(np.where(species_type==1)).sum() # How many coral species?
    
    if strategy == 'none':
        index = np.asarray([])
    elif strategy=='hot':
        index = (-temps).argsort()[0:np.int(amount*size)]
    elif strategy=='cold':
        index = temps.argsort()[0:np.int(amount*size)]
    elif strategy=='hotcold':
        index = np.r_[0:np.int(amount*size/2),np.int(size-(amount*size/2)):np.int(size)]
    elif strategy=='space':
        index = np.round(np.linspace(0,size-1,np.int(amount*size)))
    elif strategy=='highcoral':
        index = (-corals).argsort()[0:np.int(amount*size)]
    elif strategy=='lowcoral':
        index = corals.argsort()[0:np.int(amount*size)]
    elif strategy=='random':
        if subregion_only==True:
            sub_size = subregion_ID.shape[0]
            index = subregion_ID[np.random.choice(np.arange(0,sub_size),np.int(amount*sub_size), replace=False)]
        else:
            index = np.random.choice(np.arange(0,size),np.int(amount*size), replace=False)
    elif strategy == 'egvec_cent':
        sorted_centrality_array = egvec_centrality_sorted # Retrieve the site indices
        index = sorted_centrality_array[0:np.int(amount*size)]
    elif strategy == 'bw_cent':
        sorted_centrality_array = bw_centrality_sorted # Retrieve the site indices
        index = sorted_centrality_array[0:np.int(amount*size)]
    elif strategy == 'even_space':
        even_space_array = even_space
        index = even_space_array[0:np.int(amount*size)]
    
    mpa[index.astype(int)]=1
    
    return np.array([mpa]) 


#! CONTINUOUS TEMPERATURE FUNCTION
def temperature_fun(step_size, SST_matrix, tick):
    T  = np.int(tick+step_size)
    if (T<1):
        SST = SST_matrix[:,T+1]
    elif ((T+1) < SST_matrix.shape[1]):
        SST = SST_matrix[:,T]*(1-step_size) + SST_matrix[:,T+1]*(step_size)
    else:
        SST = SST_matrix[:,T-1]
    return SST

#! CONTINUOUS ALGAL MORTALITY FUNCTION
def algaemort_fun(step_size, algmort_matrix, tick):
    T  = np.int(tick+step_size)
    if (T<1):
        algmort = algmort_matrix[:,T+1]
    elif ((T+1) < algmort_matrix.shape[1]):
        algmort = algmort_matrix[:,T]*(1-step_size) + algmort_matrix[:,T+1]*(step_size)
    else:
        algmort = algmort_matrix[:,T-1]
    return algmort
            
#! INTEGRATION FUNCTION       
def coral_trait_stoch_fun(param,spp_state,trait_state,SST_matrix,algaemort_full):
   
    nsp = param['nsp']
    size = param['size']
    time_steps = param['time_steps']
    species_type = param['species_type']
    r_max = param['r_max']
    V = param['V']
    D = param['D']
    beta = param['beta']
    m_const = param['m_const']
    w = param['w']
    alphas = param['alphas']
    mpa_status = param['mpa_status']
    mortality_model = param['mortality_model']
    timemod = param['timemod']
    areas = param['areas']

    N_ALL = []
    Z_ALL = []
    YEAR = []
    N_ALL.append(spp_state)
    Z_ALL.append(trait_state)
    YEAR.append(0)
    
    algaemort_sub = algaemort_full[:,timemod:timemod+time_steps]
    tick = 0 #keeps track of temperature and algal mortality
    year = 0 #keeps track of the year array
    index = 0 #keeps track of the N_ALL and Z_ALL arrays

    # Second-order Runge Kutta solver
    while (year < time_steps-1):
        reef_ID = []
        alg_mort = algaemort_sub[:,tick]
        
        dN1 = dNdt_fun(r_max,SST_matrix[:,tick],Z_ALL[index],w,alphas,species_type,mpa_status,
                         N_ALL[index],m_const,mortality_model,alg_mort,V,D,beta,areas)
        dZ1 = dZdt_fun(r_max,SST_matrix[:,tick],Z_ALL[index],w,alphas,species_type,mpa_status,
                        N_ALL[index],m_const,mortality_model,alg_mort,V,D,beta,areas)

        N_ALL_1 = N_ALL[index] + dN1
        Z_ALL_1 = Z_ALL[index] + dZ1

        dN2 = dNdt_fun(r_max,SST_matrix[:,tick],Z_ALL_1,w,alphas,species_type,mpa_status,
                         N_ALL_1,m_const,mortality_model,alg_mort,V,D,beta,areas)
        dZ2 = dZdt_fun(r_max,SST_matrix[:,tick],Z_ALL_1,w,alphas,species_type,mpa_status,
                        N_ALL_1,m_const,mortality_model,alg_mort,V,D,beta,areas)
        
        dN_ave = (dN1 + dN2)/2
        dZ_ave = (dZ1 + dZ2)/2
        
        N_ALL.append(N_ALL[index]+dN_ave)
        Z_ALL.append(Z_ALL[index]+dZ_ave)

        reef_ID = np.where(N_ALL[index+1].sum(axis=1) > 1.0)[0]
        
        if len(reef_ID > 0):
            free_space = (1-N_ALL[index].sum(axis=1))[reef_ID]
            ID = np.where(free_space < 0.)
            free_space[ID] = 0.
            step_size0 = free_space / (dN_ave.sum(axis=1)[reef_ID])
            step_size = min(step_size0)
            if step_size <= 0:
                N_ALL[-1][reef_ID] = N_ALL[index-1][reef_ID]
                Z_ALL[-1][reef_ID] = Z_ALL[index-1][reef_ID]
                step_size = 1
            else: #if step_size is within a reasonable range
                N_ALL[-1]= N_ALL[index] + step_size*(dN1 + dN2)/2
                Z_ALL[-1]= Z_ALL[index] + step_size*(dZ1 + dZ2)/2
       
        else:
            step_size = 1 #for accounting purposes
            
        year = year + step_size
        tick = int(year)
        index += 1
    
        YEAR.append(year)
        
    return N_ALL, Z_ALL, YEAR

#! MAIN ROUTINE
def run_full_sim(P):

    #! HINDCAST
    size = P.D.shape[0] # Number of reefs in region
    total_duration = P.SST_45.shape[1]
    hindcast = P.hindcast_length #length of observed hindcast
    forecast = total_duration-hindcast #length of forecast
    SST_hindcast = P.SST_45[:,0:hindcast] #RCP 4.5 and 8.5 have the same hindcast
    SST_forecast1 = P.SST_45[:,hindcast:hindcast+forecast]
    SST_forecast2 = P.SST_85[:,hindcast:hindcast+forecast]
    
    algaemort = np.random.uniform(P.algmort_min,P.algmort_max,(total_duration)*size).reshape((size,total_duration))

    mid = SST_hindcast[:,0].mean()
    temp_range = SST_hindcast[:,0].max() - SST_hindcast[:,0].min()
    SST0 = SST_hindcast[:,0:20].mean(axis=1)
    spp_state = generate_state_fun(size, P.nsp, cover=0.25,random=False)
    trait_state = generate_traits_fun(P.nsp,size,SST0,mid,temp_range,trait_scenario='perfect_adapt')
    
    
    parameters_mpa_dict = {'temps':SST0,
                            'N_all':spp_state,
                            'species_type':P.species_type,
                            'size':size,
                            'D':P.D,
                            'subregion_ID':P.subregion_ID,
                            'egvec_centrality_sorted':P.egvec_centrality_sorted,
                            'bw_centrality_sorted':P.bw_centrality_sorted,
                            'even_space':P.even_space
                            }
    
    mpa_status = set_MPA_fun(parameters_mpa_dict,amount=0,strategy='none', subregion_only=P.subregion_flag)
    
    time_steps = hindcast
    timemod = 0 #to offset algae mortality index
    parameters_dict = {'nsp': P.nsp, 
                        'size': size, 
                        'time_steps': hindcast, 
                        'species_type': P.species_type, 
                        'V': P.V, 
                        'D': P.D, 
                        'beta': P.beta,
                        'r_max': P.r_max,
                        'alphas': P.alphas,
                        'mortality_model': P.mortality_model,
                        'mpa_status': mpa_status,
                        'w': P.w,
                        'm_const': P.m_const,
                        'timemod': timemod,
                        'areas': P.areas
                        }
    N_0, Z_0, YEAR_0 = coral_trait_stoch_fun(parameters_dict,spp_state,trait_state,SST_hindcast,algaemort)

    #! FORECAST 1: RCP 4.5
    #Can alter this; this is what determines the MPA (for strategies that depend on temp)
    SST1 = SST_hindcast[:,0:20].mean(axis=1)
    spp_state = N_0[-1][:][:]
    trait_state = Z_0[-1][:][:]
    timemod = hindcast #to offset algae mortality index
    time_steps = forecast

    #! FORECAST: with or without MPAs
    # Might need to change this depending on the particular scenario
    
    parameters_mpa_dict = {'temps':SST1,
                        'N_all':spp_state,
                        'species_type':P.species_type,
                        'size':size,
                        'D':P.D,
                        'subregion_ID':P.subregion_ID,
                        'egvec_centrality_sorted':P.egvec_centrality_sorted,
                        'bw_centrality_sorted':P.bw_centrality_sorted,
                        'even_space':P.even_space
                        }

    mpa_status = set_MPA_fun(parameters_mpa_dict,amount=P.reserve_fraction,strategy=P.reserve_strategy,subregion_only=P.subregion_flag)

    parameters_dict = {'nsp': P.nsp, 
                        'size': size, 
                        'time_steps': forecast, 
                        'species_type': P.species_type, 
                        'V': P.V, 
                        'D': P.D, 
                        'beta': P.beta,
                        'r_max': P.r_max,
                        'alphas': P.alphas,
                        'mortality_model': P.mortality_model,
                        'mpa_status': mpa_status,
                        'w': P.w,
                        'm_const': P.m_const,
                        'timemod': timemod,
                        'areas': P.areas
                        }
    N_1, Z_1, YEAR_1 = coral_trait_stoch_fun(parameters_dict,spp_state,trait_state,SST_forecast1,algaemort)
    
    #! FORECAST 2: RCP 8.5
    #Can alter this; this is what determines the MPA (for strategies that depend on temp)
    SST1 = SST_hindcast[:,0:20].mean(axis=1)
    spp_state = N_0[-1][:][:]
    trait_state = Z_0[-1][:][:]
    timemod = hindcast #to offset algae mortality index
    time_steps = forecast

    #! FORECAST: with or without MPAs
    parameters_mpa_dict = {'temps':SST1,
                    'N_all':spp_state,
                    'species_type':P.species_type,
                    'size':size,
                    'D':P.D,
                    'subregion_ID':P.subregion_ID,
                    'egvec_centrality_sorted':P.egvec_centrality_sorted,
                    'bw_centrality_sorted':P.bw_centrality_sorted,
                    'even_space':P.even_space
                    }

    mpa_status = set_MPA_fun(parameters_mpa_dict,amount=P.reserve_fraction,strategy=P.reserve_strategy,subregion_only=P.subregion_flag)

    parameters_dict = {'nsp': P.nsp, 
                        'size': size, 
                        'time_steps': forecast, 
                        'species_type': P.species_type, 
                        'V': P.V, 
                        'D': P.D, 
                        'beta': P.beta,
                        'r_max': P.r_max,
                        'alphas': P.alphas,
                        'mortality_model': P.mortality_model,
                        'mpa_status': mpa_status,
                        'w': P.w,
                        'm_const': P.m_const,
                        'timemod': timemod,
                        'areas': P.areas
                        }
    N_2, Z_2, YEAR_2 = coral_trait_stoch_fun(parameters_dict,spp_state,trait_state,SST_forecast2,algaemort)
    
    return N_0, Z_0, YEAR_0, N_1, Z_1, YEAR_1, N_2, Z_2, YEAR_2, mpa_status
    
    
#! MAIN ROUTINE: HINDCAST ONLY!
def run_hindcast(P):

    #! HINDCAST
    size = P.D.shape[0] # Number of reefs in region
    total_duration = P.SST_45.shape[1]
    hindcast = P.hindcast_length #length of observed hindcast
    forecast = total_duration-hindcast #length of forecast
    SST_hindcast = P.SST_45[:,0:hindcast] #RCP 4.5 and 8.5 have the same hindcast
    SST_forecast1 = P.SST_45[:,hindcast:hindcast+forecast]
    SST_forecast2 = P.SST_85[:,hindcast:hindcast+forecast]
    
    algaemort = np.random.uniform(P.algmort_min,P.algmort_max,(total_duration)*size).reshape((size,total_duration))

    mid = SST_hindcast[:,0].mean()
    temp_range = SST_hindcast[:,0].max() - SST_hindcast[:,0].min()
    SST0 = SST_hindcast[:,0:20].mean(axis=1)
    spp_state = generate_state_fun(size, P.nsp, cover=0.25,random=False)
    trait_state = generate_traits_fun(P.nsp,size,SST0,mid,temp_range,trait_scenario='perfect_adapt')
    parameters_mpa_dict = {'temps':SST0,
                        'N_all':spp_state,
                        'species_type':P.species_type,
                        'size':size,
                        'D':P.D,
                        'subregion_ID':P.subregion_ID,
                        'egvec_centrality_sorted':P.egvec_centrality_sorted,
                        'bw_centrality_sorted':P.bw_centrality_sorted,
                        'even_space':P.even_space
                        }
    mpa_status = set_MPA_fun(parameters_mpa_dict,amount=0,strategy='none',subregion_only=P.subregion_flag)
                    
    time_steps = hindcast
    timemod = 0 #to offset algae mortality index
    parameters_dict = {'nsp': P.nsp, 
                        'size': size, 
                        'time_steps': hindcast, 
                        'species_type': P.species_type, 
                        'V': P.V, 
                        'D': P.D, 
                        'beta': P.beta,
                        'r_max': P.r_max,
                        'alphas': P.alphas,
                        'mortality_model': P.mortality_model,
                        'mpa_status': mpa_status,
                        'w': P.w,
                        'm_const': P.m_const,
                        'timemod': timemod,
                        'areas': P.areas
                        }
    N_0, Z_0, YEAR_0 = coral_trait_stoch_fun(parameters_dict,spp_state,trait_state,SST_hindcast,algaemort)

    return N_0, Z_0, YEAR_0
    
    
#! MAIN ROUTINE: FORECAST ONLY!
def run_forecast(P,V):

    #! HINDCAST PARAMETERS
    size = P.D.shape[0] # Number of reefs in region
    total_duration = P.SST_45.shape[1]
    hindcast = P.hindcast_length #length of observed hindcast
    forecast = total_duration-hindcast #length of forecast
    SST_hindcast = P.SST_45[:,0:hindcast] #RCP 4.5 and 8.5 have the same hindcast
    SST_forecast1 = P.SST_45[:,hindcast:hindcast+forecast]
    SST_forecast2 = P.SST_85[:,hindcast:hindcast+forecast]
    
    algaemort = np.random.uniform(P.algmort_min,P.algmort_max,(total_duration)*size).reshape((size,total_duration))

    #! FORECAST 1: RCP 4.5
    #Can alter this; this is what determines the MPA (for strategies that depend on temp)
    SST1 = SST_hindcast[:,0:20].mean(axis=1)
    spp_state = P.N_0[-1][:][:]
    trait_state = P.Z_0[-1][:][:]
    timemod = hindcast #to offset algae mortality index
    time_steps = forecast

    #! FORECAST: with or without MPAs
    # Might need to change this depending on the particular scenario
    parameters_mpa_dict = {'temps':SST1,
                    'N_all':spp_state,
                    'species_type':P.species_type,
                    'size':size,
                    'D':P.D,
                    'subregion_ID':P.subregion_ID,
                    'egvec_centrality_sorted':P.egvec_centrality_sorted,
                    'bw_centrality_sorted':P.bw_centrality_sorted,
                    'even_space':P.even_space
                    }

    mpa_status = set_MPA_fun(parameters_mpa_dict,amount=P.reserve_fraction,strategy=P.reserve_strategy,subregion_only=P.subregion_flag)
        
    parameters_dict = {'nsp': P.nsp, 
                        'size': size, 
                        'time_steps': forecast, 
                        'species_type': P.species_type, 
                        'V': np.array([V,V,V]), 
                        'D': P.D, 
                        'beta': P.beta,
                        'r_max': P.r_max,
                        'alphas': P.alphas,
                        'mortality_model': P.mortality_model,
                        'mpa_status': mpa_status,
                        'w': P.w,
                        'm_const': P.m_const,
                        'timemod': timemod,
                        'areas': P.areas
                        }
    N_1, Z_1, YEAR_1 = coral_trait_stoch_fun(parameters_dict,spp_state,trait_state,SST_forecast1,algaemort)
    
    #! FORECAST 2: RCP 8.5
    #Can alter this; this is what determines the MPA (for strategies that depend on temp)
    SST1 = SST_hindcast[:,0:20].mean(axis=1)
    spp_state = P.N_0[-1][:][:]
    trait_state = P.Z_0[-1][:][:]
    timemod = hindcast #to offset algae mortality index
    time_steps = forecast

    #! FORECAST: with or without MPAs
    parameters_mpa_dict = {'temps':SST1,
                    'N_all':spp_state,
                    'species_type':P.species_type,
                    'size':size,
                    'D':P.D,
                    'subregion_ID':P.subregion_ID,
                    'egvec_centrality_sorted':P.egvec_centrality_sorted,
                    'bw_centrality_sorted':P.bw_centrality_sorted,
                    'even_space':P.even_space
                    }

    mpa_status = set_MPA_fun(parameters_mpa_dict,amount=P.reserve_fraction,strategy=P.reserve_strategy,subregion_only=P.subregion_flag)
        
    parameters_dict = {'nsp': P.nsp, 
                        'size': size, 
                        'time_steps': forecast, 
                        'species_type': P.species_type, 
                        'V': np.array([V,V,V]), 
                        'D': P.D, 
                        'beta': P.beta,
                        'r_max': P.r_max,
                        'alphas': P.alphas,
                        'mortality_model': P.mortality_model,
                        'mpa_status': mpa_status,
                        'w': P.w,
                        'm_const': P.m_const,
                        'timemod': timemod,
                        'areas': P.areas
                        }
    N_2, Z_2, YEAR_2 = coral_trait_stoch_fun(parameters_dict,spp_state,trait_state,SST_forecast2,algaemort)
    
    return N_1, Z_1, YEAR_1, N_2, Z_2, YEAR_2, mpa_status