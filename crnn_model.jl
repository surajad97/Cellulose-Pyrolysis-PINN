#Random.seed!()
np = nr * (ns + nc + 3) + 1;
p = randn(Float64, np) .* 1.e-2;
p[1:nr] .+= 0.8;  # w_lnA
#p[nr*(nss+1)+1:nr*(ns+1)].*= 10            #w_cat_in
p[nr*(ns+1)+1:nr*(ns+nc+1)] .= 0           #w_cat_out
p[nr*(ns+nc+1)+1:nr*(ns'+nc+2)] .+= 0.6;  # w_Ea
p[nr*(ns+nc+2)+1:nr*(ns+nc+3)] .+= 0.1;  # w_b
p[end] = 0.1;  #slope

function p2vec(p)
    slope = p[end] .* 1.e1
    w_b = p[1:nr] .* (slope * 10.0)
    w_b = clamp.(w_b, 0, 50)

    w_out = reshape(p[nr+1:nr*(nss+1)], nss, nr)
    @. w_out[1, :] = clamp(w_out[1, :], -3.0, 0.0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0.0, 3.0)

    if p_cutoff > 0.0
        w_out[findall(abs.(w_out) .< p_cutoff)] .= 0.0
    end

    w_out[nss-1:nss-1, :] .=
        -sum(w_out[1:nss-2, :], dims = 1) .- sum(w_out[nss:nss, :], dims = 1)

    w_cat_in = p[nr*(nss+1)+1:nr*(ns+1)]
    w_cat_out = p[nr*(ns+1)+1:nr*(ns+nc+1)]*0
    
    #w_cat_out = - sign.(w_cat_in) .* abs.(w_cat_out);
    w_cat_in = abs.(w_cat_in)
    
    if p_cutoff > 0.0
        w_cat_in[findall(abs.(w_cat_in) .< p_cutoff)] .= 0.0
    end
    
    w_in_Ea = abs.(p[nr*(ns+nc+1)+1:nr*(ns+nc+2)].* (slope * 100.0))
    w_in_Ea = clamp.(w_in_Ea, 0.0, 300.0)

    w_in_b = abs.(p[nr*(ns+nc+2)+1:nr*(ns+nc+3)])


    w_in = vcat(clamp.(-w_out, 0.0, 3.0), w_cat_in', w_in_Ea', w_in_b')
    w_out = vcat(w_out, w_cat_out')
    
    return w_in, w_b, w_out
end

function display_p(p)
    
    w_in, w_b, w_out = p2vec(p)
    println("\n species (column) reaction (row)")
    println("w_in | w_cat_in | Ea | b | lnA | w_out | w_cat_out")
    show(stdout, "text/plain", round.(hcat(w_in', w_b, w_out'), digits = 2))
    println("\n")
end

display_p(p)

function getsampletemp(t, T0, beta)
    T = clamp.((T0 + (beta / 60) * t), 0, 873.15)   # K/min to K/s
    return T
end

const R = -1.0 / 8.314e-3  # universal gas constant, kJ/mol*K
@inbounds function crnn!(du, u, p, t)
    logX = @. log(clamp(u, lb, 10.0))
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, R / T, log(T))
    du .= w_out * (@. exp(w_in_x + w_b))
end

tspan = [0.0, 1.0];
u0 = zeros(ns);
u0[1] = 1.0;
prob = ODEProblem(crnn!, u0, tspan, p, abstol = lb)

condition(u, t, integrator) = u[1] < lb * 10.0
affect!(integrator) = terminate!(integrator)
_cb = DiscreteCallback(condition, affect!)

alg = TRBDF2();
#alg = AutoTsit5(Rosenbrock23(autodiff=false));
#alg = Rosenbrock23(autodiff=false);
#alg = AutoVern7(Rodas5P(autodiff = false));
#sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
sense = ForwardSensitivity(autojacvec = true)

function pred_n_ode(p, i_exp, exp_data)
    global T0, beta, cat_conc = l_exp_info[i_exp, :]
    global w_in, w_b, w_out = p2vec(p)

    ts = @view(exp_data[:, 1])
    tspan = [ts[1], ts[end]]
    u0[end] = cat_conc 
    sol = solve(
        prob,
        alg,
        tspan = tspan,
        p = p,
        saveat = ts, 
        sensealg = sense,
        maxiters = maxiters
        # callback = _cb,
    )

    if sol.retcode == :Success
        nothing
    else
        #@sprintf("solver failed beta: %.0f ocen: %.2f", beta, exp(ocen))
        @sprintf("solver failed beta: %.0f", beta)
    end
    if length(sol.t) > length(ts)
        # @show exp_data[:, 1]
        # @show sol.t
        return  sol[:, 1:length(ts)]
    else
        return sol
    end
end

function loss_neuralode(p, i_exp)
    exp_data = l_exp_data[i_exp]
    pred = Array(pred_n_ode(p, i_exp, exp_data))
    masslist = sum(clamp.(@view(pred[1:end-1-nc, :]), 0, Inf), dims = 1)'
    gaslist = clamp.(@views(pred[end-nc, :]), 0,  Inf)

    loss = mae(masslist, @view(exp_data[1:length(masslist), 3])) + mae(gaslist, 1 .- @view(exp_data[1:length(masslist), 3]))
    return loss
end

@time loss = loss_neuralode(p, 1)

