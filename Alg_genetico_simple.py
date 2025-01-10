#!/usr/bin/env python
# coding: utf-8

# In[24]:


from deap import creator, base, tools, algorithms

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[25]:


from turnosenfermeria import TurnosEnfermeria
import elitism as elit


# In[ ]:





# In[47]:


def cargaDatos (ruta_archivo):
    """
    Esta función carga de forma programática los datos relativos a las preferencias de horario de cada uno de los enfermer@s del servicio de urgencias
    del hospital.

    :param ruta_archivo: Dirección de memoria en la que se encuentra el archivo con las preferencias de los enfermer@s
    :return preferencias: Preferencias de horarios de los enfermer@s del servicio de urgencias
    :return enfermeras: Nombres de los enfermer@s que componen el servicio de urgencias
    """
    
    preferencias = pd.read_csv(ruta_archivo, sep=' ')
    enfermeras = preferencias["Nombre"]

    return preferencias, enfermeras
    


# In[21]:


def CreaccionCalendarioTurnos (ruta_archivo, tamano_poblacion, seleccion, p_cruce, cruce, p_mutacion, mutacion, algoritmo, max_generaciones, mu, lambd, verbose):
    """
    Esta función crea un algoritmo genético simple, el cual evalúa a través de las distintas generaciones los distintos individuos, calendarios de turnos
    en este caso, para evaluar cual es mejor según los criterios establecidos.

    :param ruta_archivo: Ruta en la que se encuentra el archivo con las preferencias de los trabajadores del servicio
    :param tamano_poblacion: Número de individuos o calendarios que van a ser evaluados por el algoritmo genético simple.
    :param seleccion: Método de selección que se va a emplear en el algoritmo.
    :param p_cruce: Probabilidad de que tenga lugar un cruce de un punto entre los distintos individuos, calendarios.
    :param cruce: Tipo de cruce que se va a ejecutar entre los individuos.
    :param p_mutacion: Probabilidad de que tenga lugar una mutación de flip-bit entre los individuos de la población.
    :param mutacion: Tipo de mutación que se aplica a la población en curso
    :param algoritmo: Tipo de algoritmo genético que se va a ejecutar
    :param max_generaciones: Número de generaciones que se evalúan antes de encontrar la solución óptima al problema
    :param mu: Número de individuos que deben ser seleccionados en cada generación.
    :param lambd: Número de hijos que se deben producir en cada generación.
    :param verbose: Con "True" o "False" nos permite indicar si queremos mostrar por pantalla el logbook o no
    :return poblacion_final: Población compuesta por los individuos que han sobrevidio al proceso evolutivo del algoritmo.
    :return logbook: Registro sobre la evolución de la población a lo largo de las generaciones.
    
    """

    tamano_individuo = 21
    preferencias, enfermeras = cargaDatos (ruta_archivo)
    turnos = TurnosEnfermeria (enfermeras, preferencias)
    tamano_calendario = len (preferencias)
    
    toolbox=base.Toolbox()
    
    creator.create("ClaseAjusteMin", base.Fitness, weights=(-1.0,))
    creator.create("ClaseIndividuo", list, fitness=creator.ClaseAjusteMin)

    toolbox.register("ceroOrUno", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.ClaseIndividuo, toolbox.ceroOrUno, tamano_individuo*tamano_calendario)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, tamano_poblacion)
    poblacion = toolbox.population()
   

    toolbox.register("evaluate", turnos.getCoste)


    
    if seleccion == "selTournament":
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif seleccion == "selBest":
        toolbox.register("select", tools.selBest)
    elif seleccion == "selRoulette":
        toolbox.register("select", tools.selRoulette)
    elif seleccion == "selStochasticUniversalSampling":
        toolbox.register("select", tools.selStochasticUniversalSampling)
    else:
        raise ValueError(f"Tipo de selección no soportada: {seleccion}")


    if cruce == "cxOnePoint":
        toolbox.register("mate", tools.cxOnePoint)
    elif cruce == "cxTwoPoint":
        toolbox.register("mate", tools.cxTwoPoint)
    elif cruce == "cxUniform":
        toolbox.register("mate", tools.cxUniform)
    elif cruce == "cxOrdered":
        toolbox.register("mate", tools.cxOrdered)
    else:
        raise ValueError(f"Tipo de cruce no soportado: {cruce}")


    if mutacion == "mutFlipBit":
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/tamano_individuo)
    elif mutacion == "mutShuffleIndexes":
        toolbox.register("mutate", tools.mutShuffleIndexes)
    elif mutacion == "mutInversion":
        toolbox.register("mutate", tools.mutInversion)
    


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)
    stats.register("min", np.min)


    hof = tools.HallOfFame(1)

    if algoritmo == "eaSimple":
        poblacion_final, logbook= algorithms.eaSimple(poblacion, toolbox, cxpb=p_cruce, mutpb=p_mutacion, ngen=max_generaciones,stats=stats,
                                                  halloffame=hof,verbose=verbose)

    if algoritmo == "eaMuPlusLambda":
        poblacion_final, logbook = algorithms.eaMuPlusLambda(poblacion, toolbox, mu=mu, lambd=lambd, cxpb=p_cruce, mutpb=p_mutacion, ngen=max_generaciones,                                                       stats=stats, halloffame=hof,verbose=verbose) 

    if algoritmo == "eaSimpleWithElitism":
        poblacion_final, logbook= elit.eaSimpleWithElitism(poblacion, toolbox, cxpb=p_cruce, mutpb=p_mutacion, ngen=max_generaciones,stats=stats,
                                                  halloffame=hof,verbose=verbose)
        
        

    
    turnos.mostrarInfoCalendario (hof[0])
    
    return poblacion_final, logbook


# In[ ]:

def plot_evolucion(log, titulo="Evolución de Descriptores vs Generaciones"):

    

    """
    Esta función genera un gráfico que muestra la evolución de los valores de ajuste (mínimo, máximo y medio), así como la dispersión
    a lo largo de las generaciones del un algoritmo genético a partir del obtejo Logbook del mismo.

    :param log: Objeto logbook que contiene los datos de las generaciones y valores de fitness registrados.
    :param titulo: Título del gráfico que se mostrará en la parte superior, con un valor por defecto.
    
    """
    gen=log.select("gen")
    fit_mins=log.select("min")
    fit_maxs= log.select("max")
    fit_means=log.select("avg")
    
    fig,ax = plt.subplots()
    
    ax.plot(gen,fit_mins,color="green")
    ax.plot(gen,fit_maxs,color="red")
    ax.plot(gen,fit_means,linestyle="--", color="blue")
    fit_mins=np.array(fit_mins)
    fit_maxs=np.array(fit_maxs)   
    ax.fill_between(gen,fit_mins,fit_maxs,where=(fit_maxs>=fit_mins), facecolor='y', alpha=0.2)
    
    ax.set_xlabel("Generación")
    ax.set_ylabel("Ajuste (fitness)")
    ax.set_ylim([0,110])
    ax.legend(["Min","Max","Media"])
    ax.set_title(titulo)
    plt.grid(True)





# In[ ]:




