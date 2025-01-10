#!/usr/bin/env python
# coding: utf-8

# In[24]:


from deap import creator, base, tools, algorithms

import random
import numpy as np
import pandas as pd


# In[25]:


from turnosenfermeria import TurnosEnfermeria


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


def CreaccionCalendarioTurnos (tamano_individuo, tamano_calendario, tamano_poblacion, seleccion="SelTournament", p_cruce, cruce, p_mutacion, mutacion, algoritmo, max_generaciones, verbose=True):
    """
    Esta función crea un algoritmo genético simple, el cual evalúa a través de las distintas generaciones los distintos individuos, calendarios de turnos
    en este caso, para evaluar cual es mejor según los criterios establecidos.

    :param tamano_individuo: Cadena de 21 bits repartida en 7 codones de 3 bits, que equivalen a la jornada semanal de cada uno de los enfermer@s.
    :param tamano_calendario: Filas del calendario, cada una de las cuales equivale a uno de los enfermer@s del servicio.
    :param tamano_poblacion: Número de individuos o calendarios que van a ser evaluados por el algoritmo genético simple.
    :param p_cruce: Probabilidad de que tenga lugar un cruce de un punto entre los distintos individuos, calendarios.
    :param p_mutacion: Probabilidad de que tenga lugar una mutación de flip-bit entre los individuos de la población.
    :param max_generaciones: Número de generaciones que se evalúan antes de encontrar la solución óptima al problema
    :return poblacion_final: Población compuesta por los individuos que han sobrevidio al proceso evolutivo del algoritmo.
    :return logbook: Registro sobre la evolución de la población a lo largo de las generaciones.
    
    """
    toolbox=base.Toolbox()
    creator.create("ClaseAjusteMin", base.Fitness, weights=(-1.0,))
    creator.create("ClaseIndividuo", list, fitness=creator.ClaseAjusteMin)

    toolbox.register("ceroOrUno", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.ClaseIndividuo, toolbox.ceroOrUno, tamano_individuo*tamano_calendario)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, tamano_poblacion)
    poblacion = toolbox.population()
   

    toolbox.register("evaluate", turnos.getCoste)


    
    if seleccion == "selTorunament":
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
        toolbox.register("mutate", tools.mutFlipBit)
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
    poblacion_final, logbook= algorithms.eaSimple(poblacion, toolbox, cxpb=p_cruce, mutpb=p_mutacion, ngen=max_generaciones,stats=stats,
                                                  halloffame=hof,verbose=verbose)

    
    turnos.mostrarInfoCalendario (hof[0])
    
    return poblacion_final, logbook


# In[ ]:





# In[ ]:




