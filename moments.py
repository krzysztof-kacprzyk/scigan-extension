import numpy as np
import scipy.stats as stats

num_dosage_samples = 100



a = 5
b = 5

errors_mean = []
errors_var = []

for i in range(1000):

    xs = np.random.uniform(0., 1., num_dosage_samples)

    ys = np.array([stats.beta.pdf(x,a,b) for x in xs])

    sample_mean = (xs*ys).sum()/(ys.sum())
    sample_var = (np.square(xs-sample_mean)*ys).sum()/(ys.sum())

    errors_mean.append(np.square(sample_mean - stats.beta.mean(a,b)))
    errors_var.append(np.square(sample_var - stats.beta.var(a,b)))

print(f"Average error for mean: {np.sqrt(np.mean(errors_mean))} | {100*np.mean(errors_mean)/stats.beta.mean(a,b)}%")
print(f"Average error for variance: {np.sqrt(np.mean(errors_var))} | {100*np.mean(errors_var)/stats.beta.var(a,b)}%")

# print(f"Sample mean: {sample_mean}")
print(f"True mean: {stats.beta.mean(a,b)}")

# print(f"Sample var: {sample_var}")
print(f"True var: {stats.beta.var(a,b)}")
