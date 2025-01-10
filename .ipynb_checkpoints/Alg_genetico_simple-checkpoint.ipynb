{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b6ab54c1-3e08-4073-b500-9ce604c23dbc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from deap import creator, base, tools, algorithms\n",
    "\n",
    "import random\n",
    "import numpy as np\n",
    "import pandas as pd\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c56e2b5b-16a2-42de-807c-35b5afb25401",
   "metadata": {},
   "outputs": [],
   "source": [
    "from turnosenfermeria import TurnosEnfermeria"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64c1a96b-2102-43df-8a00-749001b4135d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "6e2fdea6-b10a-4eff-b4e6-629b3893af38",
   "metadata": {},
   "outputs": [],
   "source": [
    "def cargaDatos (ruta_archivo):\n",
    "    \"\"\"\n",
    "    Esta función carga de forma programática los datos relativos a las preferencias de horario de cada uno de los enfermer@s del servicio de urgencias\n",
    "    del hospital.\n",
    "\n",
    "    :param ruta_archivo: Dirección de memoria en la que se encuentra el archivo con las preferencias de los enfermer@s\n",
    "    :return preferencias: Preferencias de horarios de los enfermer@s del servicio de urgencias\n",
    "    :return enfermeras: Nombres de los enfermer@s que componen el servicio de urgencias\n",
    "    \"\"\"\n",
    "    \n",
    "    preferencias = pd.read_csv(ruta_archivo, sep=' ')\n",
    "    enfermeras = preferencias[\"Nombre\"]\n",
    "\n",
    "    return preferencias, enfermeras\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "53af812a-8af2-4a40-987f-7e87cc5aff14",
   "metadata": {},
   "outputs": [],
   "source": [
    "def CreaccionCalendarioTurnos (tamano_individuo, tamano_calendario, tamano_poblacion, p_cruce, p_mutacion, max_generaciones, verbose=True):\n",
    "    \"\"\"\n",
    "    Esta función crea un algoritmo genético simple, el cual evalúa a través de las distintas generaciones los distintos individuos, calendarios de turnos\n",
    "    en este caso, para evaluar cual es mejor según los criterios establecidos.\n",
    "\n",
    "    :param tamano_individuo: Cadena de 21 bits repartida en 7 codones de 3 bits, que equivalen a la jornada semanal de cada uno de los enfermer@s.\n",
    "    :param tamano_calendario: Filas del calendario, cada una de las cuales equivale a uno de los enfermer@s del servicio.\n",
    "    :param tamano_poblacion: Número de individuos o calendarios que van a ser evaluados por el algoritmo genético simple.\n",
    "    :param p_cruce: Probabilidad de que tenga lugar un cruce de un punto entre los distintos individuos, calendarios.\n",
    "    :param p_mutacion: Probabilidad de que tenga lugar una mutación de flip-bit entre los individuos de la población.\n",
    "    :param max_generaciones: Número de generaciones que se evalúan antes de encontrar la solución óptima al problema\n",
    "    :return poblacion_final: Población compuesta por los individuos que han sobrevidio al proceso evolutivo del algoritmo.\n",
    "    :return logbook: Registro sobre la evolución de la población a lo largo de las generaciones.\n",
    "    \n",
    "    \"\"\"\n",
    "    toolbox=base.Toolbox()\n",
    "    creator.create(\"ClaseAjusteMin\", base.Fitness, weights=(-1.0,))\n",
    "    creator.create(\"ClaseIndividuo\", list, fitness=creator.ClaseAjusteMin)\n",
    "\n",
    "    toolbox.register(\"ceroOrUno\", random.randint, 0, 1)\n",
    "    toolbox.register(\"individual\", tools.initRepeat, creator.ClaseIndividuo, toolbox.ceroOrUno, tamano_individuo*tamano_calendario)\n",
    "    toolbox.register(\"population\", tools.initRepeat, list, toolbox.individual, tamano_poblacion)\n",
    "    poblacion = toolbox.population()\n",
    "   \n",
    "\n",
    "\n",
    "    toolbox.register(\"evaluate\", turnos.getCoste)\n",
    "\n",
    "\n",
    "    toolbox.register(\"select\", tools.selTournament, tournsize=3)\n",
    "    toolbox.register(\"mate\", tools.cxOnePoint)\n",
    "    toolbox.register(\"mutate\", tools.mutFlipBit, indpb=1.0/tamano_individuo)\n",
    "\n",
    "\n",
    "    stats = tools.Statistics(lambda ind: ind.fitness.values)\n",
    "    stats.register(\"avg\", np.mean)\n",
    "    stats.register(\"std\", np.std)\n",
    "    stats.register(\"max\", np.max)\n",
    "    stats.register(\"min\", np.min)\n",
    "\n",
    "\n",
    "    hof = tools.HallOfFame(1)\n",
    "    poblacion_final, logbook= algorithms.eaSimple(poblacion, toolbox, cxpb=p_cruce, mutpb=p_mutacion, ngen=max_generaciones,stats=stats,\n",
    "                                                  halloffame=hof,verbose=verbose)\n",
    "\n",
    "    \n",
    "    turnos.mostrarInfoCalendario (hof[0])\n",
    "    \n",
    "    return poblacion_final, logbook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82e522d0-c9fe-43c3-a941-e025a0507c88",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b4c60ff-5fac-445e-8585-0884c1c89c99",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
