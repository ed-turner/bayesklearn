# this defines the data we will use
def assemble_data(cat_cols, is_regression=True):
    """
    Inputs:
        numeric_cols - a list of our numeric columns
        target_cols - our target column
        cat_cols - a list of tuples for our categorial variables in this format, (cat_col, num_unique_cat_values)
    """

    # we get our input sizes
    feature_sizes = 'int<lower = 1> N; int<lower = 1> p; '

    # we assemble our vector variables
    features = feature_sizes + 'matrix[N, p] x; '

    if is_regression:
        features += 'vector[N] y; real<lower=0> nu; '
    else:
        features += 'int<lower=0, upper=1> y[N]; real<lower=0> nu;'

    # we return our vector variables if we do not have any category columns
    if len(cat_cols) == 0:
        features += 'real y_mean; real y_std; '
        return features

    # else, we return the effects for each categorial feature
    else:
        categorials = ['int ' + cat_col[0] + '[N]' for cat_col in cat_cols]
        categorials += ['vector[' + cat_col[1] + '] y_mean_' + cat_col[0] for cat_col in cat_cols]
        categorials += ['vector[' + cat_col[1] + '] y_std_' + cat_col[0] for cat_col in cat_cols]

        return features + '; '.join(categorials) + '; '


# effects
def assemble_category_effects(cat_cols):
    means = ['real mu_beta_' + cat_col[0] for cat_col in cat_cols]
    sigma = ['real<lower=0> sigma_beta_' + cat_col[0] for cat_col in cat_cols]
    effects = ['vector[' + cat_col[1] + '] effect_' + cat_col[0] for cat_col in cat_cols]

    return '; '.join(means + sigma + effects) + '; '


# this returns a string of the parameters data structure and a list of all the parameters
def assemble_parameters(cat_cols, is_regression):
    '''
    Inputs :
        cat_cols -  the categorial columns, a list of tuples defined similarily as the assembleData method
    '''

    # if there were not any categorial faetures, we just return the beta parameter and the sigma parameter
    if len(cat_cols) == 0:
        params = 'vector[p] beta; real<lower = 0> sigma; real shift; real mu_b; real<lower=0> sigma_b; real mu_s; ' \
                 'real<lower=0> sigma_s; '

        if is_regression:
            pass
        else:
            params += 'vector[N] eta; '

        return params

    # these are for the shifts.. we may set this as a separate variable until we figure out the beta problem
    means = ['real mu_' + cat_col[0] for cat_col in cat_cols] + ['real mu_b']
    sigmas = ['real<lower=0> sigma_' + cat_col[0] for cat_col in cat_cols] + ['real<lower=0> sigma_b',
                                                                              'real<lower=0> sigma']

    # we assume the weights are all the same for now
    betas = ['vector[p] beta']
    shifts = ['vector[' + cat_col[1] + '] shift_' + cat_col[0] for cat_col in cat_cols]

    params = '; '.join(means + sigmas + betas + shifts) + '; '

    if is_regression:
        pass
    else:
        params += 'vector[N] eta; '

    return params


# this assembles the model depending on whether we are performing regression
def assemble_model_code_gaussian_regression(cat_cols):
    if len(cat_cols) == 0:
        weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 20); beta ~ normal(mu_b, sigma_b); '
        shifts_dis = 'mu_s ~ normal(y_mean, y_std); sigma_s ~ normal(0,2); shift ~ normal(mu_s, sigma_s); '

        model = 'sigma ~ normal(0, 20); y ~ student_t(nu, x * beta + shift, sigma ); '

    else:

        # this is a helper function
        def setupShiftDist(col):
            mu_dist = 'mu_' + col + ' ~ normal(y_mean_' + col + ', y_std_' + col + '); '
            sigma_dist = 'sigma_' + col + ' ~ normal(0, 2); '
            return mu_dist + sigma_dist + 'shift_' + col + ' ~ normal(mu_' + col + ', sigma_' + col + '); '

        shifts_dis = ' '.join([setupShiftDist(cat_col[0]) for cat_col in cat_cols])
        weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 20); beta ~ normal(mu_b, sigma_b); '

        shifts = '+ '.join(['shift_' + cat_col[0] + '[' + cat_col[0] + ']' for cat_col in cat_cols])
        weighted = 'x*beta'

        model = 'sigma ~ normal(0, 20); y ~ student_t(nu, ' + weighted + ' + ' + shifts + ', sigma); '

    return weight_dis + shifts_dis + model


# this is gaussian logistic regression
def assemble_model_code_gaussian_classification(cat_cols):
    if len(cat_cols) == 0:
        weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 5); beta ~ normal(mu_b, sigma_b); '
        shifts_dis = 'mu_s ~ normal(y_mean, y_std); sigma_s ~ normal(0,2); shift ~ normal(mu_s, sigma_s); '

        model = 'sigma ~ normal(0, 20); eta ~ normal( x * beta + shift, sigma ); y ~ bernoulli_logit( eta ); '

    else:
        # this is a helper function
        def setup_shift_dist(col):
            mu_dist = 'mu_' + col + ' ~ normal(y_mean_' + col + ', y_std_' + col + '); '
            sigma_dist = 'sigma_' + col + ' ~ normal(0, 2); '
            return mu_dist + sigma_dist + 'shift_' + col + ' ~ normal(mu_' + col + ', sigma_' + col + '); '

        shifts_dis = ' '.join([setup_shift_dist(cat_col[0]) for cat_col in cat_cols])
        weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 20); beta ~ normal(mu_b, sigma_b); '

        shifts = '+ '.join(['shift_' + cat_col[0] + '[' + cat_col[0] + ']' for cat_col in cat_cols])
        eqn = 'x*beta + ' + shifts

        model = 'sigma ~ normal(0, 20); eta ~ student_t(nu,' + eqn + ', sigma); y ~ bernoulli_logit(eta); '

    return weight_dis + shifts_dis + model


# this returns the model code
def assemble_model_code(cat_cols=None, is_regression=True):
    if cat_cols is None:
        cat_cols = []
    data = assemble_data(cat_cols, is_regression=is_regression)

    params = assemble_parameters(cat_cols, is_regression)

    if is_regression:
        model = assemble_model_code_gaussian_regression(cat_cols)

    else:
        model = assemble_model_code_gaussian_classification(cat_cols)

    return 'data {' + data + '} parameters {' + params + '} model {' + model + '}'
