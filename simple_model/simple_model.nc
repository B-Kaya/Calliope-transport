�HDF

         ��������6�     0       �4eOHDRJ"     ,       a�      $     9$     
          �      	      >	       �                                                                                                                                                                                                                                               �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                �`x�FRHP              �      �0      d	       @              (             ͠                                            (  b�       ��1�BTHD       d(�	              Io�BTHD 	      d(�              ���FSHD  �	                            P x (        �      D       D       ��m�BTLF  �'   R  
   �d  {    <     ��          j�    O     >��  �&   @     �*% 3   E     l3�-    1     +�
3 H    3     ��!6    �    �Q@     y    kW�G    (     Ce`     2      @�k �&   R     �s� �    (     ʻ#� P   R     kϬ� �&   $     ��� �   .     Ŵ�� �   2     �]Y� u   r     f�� E'   O  	   1��� �    "     ��N� 5   @     K�����j�                                                                                                                                BTLF 	     2       H    3      {    <      �    (      �    "          y     �&   @      �&   $      �&   R      E'   O  	    �'   R  
       O      P   R                   �        (      5   @      u   r      �   2      �   .         1      3   E     a�ݽ                                                                                                                                                                                                                        BTHD       d(��              �*�i     applied_math    ��     history:
- plan
data:
  variables:
    flow_cap:
      title: Technology flow (a.k.a. nominal) capacity
      description: A technology's flow capacity, also known as its nominal or nameplate
        capacity.
      default: 0
      unit: power
      foreach:
      - nodes
      - techs
      - carriers
      bounds:
        min: 0
        max: flow_cap_max
    link_flow_cap:
      title: Link flow capacity
      description: A transmission technology's flow capacity, also known as its nominal
        or nameplate capacity.
      default: 0
      unit: power
      foreach:
      - techs
      where: base_tech=transmission
      bounds:
        min: 0
        max: .inf
    flow_out:
      title: Carrier outflow
      description: The outflow of a technology per timestep, also known as the flow
        discharged (from `storage` technologies) or the flow received (by `transmission`
        technologies) on a link.
      default: 0
      unit: energy
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: carrier_out
      bounds:
        min: 0
        max: .inf
    flow_in:
      title: Carrier inflow
      description: The inflow to a technology per timestep, also known as the flow
        consumed (by `storage` technologies) or the flow sent (by `transmission` technologies)
        on a link.
      default: 0
      unit: energy
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: carrier_in
      bounds:
        min: 0
        max: .inf
    flow_export:
      title: Carrier export
      description: The flow of a carrier exported outside the system boundaries by
        a technology per timestep.
      default: 0
      unit: energy
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: carrier_export
      bounds:
        min: 0
        max: .inf
    area_use:
      title: Area utilisation
      description: The area in space utilised directly (e.g., solar PV panels) or
        indirectly (e.g., biofuel crops) by a technology.
      default: 0
      unit: area
      foreach:
      - nodes
      - techs
      where: (area_use_min OR area_use_max OR area_use_per_flow_cap OR sink_unit=per_area
        OR source_unit=per_area)
      bounds:
        min: 0
        max: area_use_max
    source_use:
      title: Source flow use
      description: The carrier flow consumed from outside the system boundaries by
        a `supply` technology.
      default: 0
      unit: energy
      foreach:
      - nodes
      - techs
      - timesteps
      where: base_tech=supply
      bounds:
        min: 0
        max: .inf
    source_cap:
      title: Source flow capacity
      description: The upper limit on a flow that can be consumed from outside the
        system boundaries by a `supply` technology in each timestep.
      default: 0
      unit: power
      foreach:
      - nodes
      - techs
      where: base_tech=supply
      bounds:
        min: 0
        max: source_cap_max
    storage_cap:
      title: Stored carrier capacity
      description: The upper limit on a carrier that can be stored by a technology
        in any timestep.
      default: 0
      unit: energy
      foreach:
      - nodes
      - techs
      where: include_storage=True OR base_tech=storage
      domain: real
      bounds:
        min: 0
        max: storage_cap_max
      active: true
    storage:
      title: Stored carrier
      description: The carrier stored by a `storage` technology in each timestep.
      default: 0
      unit: energy
      foreach:
      - nodes
      - techs
      - timesteps
      where: include_storage=True OR base_tech=storage
      bounds:
        min: 0
        max: .inf
    purchased_units:
      title: Number of purchased units
      description: "Integer number of a technology that has been purchased,\nfor any
        technology set to require integer capacity purchasing.\nThis is used to allow
        installation of fixed capacity units of technologies (\nif `flow_cap_max`
        == `flow_cap_min`) and/or to set a fixed cost for a technology,\nirrespective
        of its installed capacity.\nOn top of a fixed technology cost,\na continuous
        cost for the quantity of installed capacity can still be applied.\n\nSince
        technology capacity is no longer a continuous decision variable,\nit is possible
        for these technologies to have a lower bound set on outflow/consumption\n
        which will only be enforced in those timesteps that the technology is operating.\n
        Otherwise, the same lower bound forces the technology to produce/consume\n
        that minimum amount of carrier in *every* timestep.\n"
      default: 0
      unit: integer
      foreach:
      - nodes
      - techs
      where: cap_method=integer
      domain: integer
      bounds:
        min: purchased_units_min
        max: purchased_units_max
    operating_units:
      title: Number of operating units
      description: Integer number of a technology that is operating in each timestep,
        for any technology set to require integer capacity purchasing.
      default: 0
      unit: integer
      foreach:
      - nodes
      - techs
      - timesteps
      where: integer_dispatch=True AND cap_method=integer
      domain: integer
      bounds:
        min: 0
        max: .inf
    available_flow_cap:
      title: Available carrier flow capacity
      description: Flow capacity that will be set to zero if the technology is not
        operating in a given timestep and will be set to the value of the decision
        variable `flow_cap` otherwise. This is useful when you want to set a minimum
        flow capacity for any technology investment, but also want to allow the model
        to decide the capacity. It is expected to only be used when `purchased_units_max
        == 1`, i.e., the `purchased_units` decision variable is binary. If `purchased_units_max
        > 1`, you may get strange results and should instead use the less flexible
        `flow_cap_per_unit`.
      default: 0
      unit: power
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: integer_dispatch=True AND flow_cap_max AND NOT flow_cap_per_unit
      bounds:
        min: 0
        max: .inf
    async_flow_switch:
      title: Asynchronous carrier flow switch
      description: Binary switch to force asynchronous outflow/consumption of technologies
        with both `flow_in` and `flow_out` defined. This ensures that a technology
        with carrier flow efficiencies < 100% cannot produce and consume a flow simultaneously
        to remove unwanted carrier from the system.
      default: 0
      unit: integer
      foreach:
      - nodes
      - techs
      - timesteps
      where: force_async_flow=True
      domain: integer
      bounds:
        min: 0
        max: 1
    unmet_demand:
      title: Unmet demand (load shedding)
      description: Virtual source of carrier flow to ensure model feasibility. This
        should only be considered a debugging rather than a modelling tool as it may
        distort the model in other ways due to the large impact it has on the objective
        function value. When present in a model in which it has been requested, it
        indicates an inability for technologies in the model to reach a sufficient
        combined supply capacity to meet demand.
      default: 0
      unit: energy
      foreach:
      - nodes
      - carriers
      - timesteps
      where: config.ensure_feasibility=True
      bounds:
        min: 0
        max: .inf
    unused_supply:
      title: Unused supply (curtailment)
      description: 'Virtual sink of carrier flow to ensure model feasibility. This
        should only be considered a debugging rather than a modelling tool as it may
        distort the model in other ways due to the large impact it has on the objective
        function value. In model results, the negation of this variable is combined
        with `unmet_demand` and presented as only one variable: `unmet_demand`. When
        present in a model in which it has been requested, it indicates an inability
        for technologies in the model to reach a sufficient combined consumption capacity
        to meet required outflow (e.g. from renewables without the possibility of
        curtailment).'
      default: 0
      unit: energy
      foreach:
      - nodes
      - carriers
      - timesteps
      where: config.ensure_feasibility=True
      bounds:
        min: -.inf
        max: 0
  global_expressions:
    flow_out_inc_eff:
      title: Carrier outflow including losses
      description: Outflows after taking efficiency losses into account.
      default: 0
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_out
      equations:
      - where: base_tech=transmission
        expression: "flow_out / (\n  flow_out_eff * flow_out_parasitic_eff *\n  flow_out_eff_per_distance
          ** distance\n)"
      - where: NOT base_tech=transmission
        expression: flow_out / (flow_out_eff * flow_out_parasitic_eff)
    flow_in_inc_eff:
      title: Carrier inflow including losses
      description: Inflows after taking efficiency losses into account.
      default: 0
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_in
      equations:
      - where: base_tech=transmission
        expression: flow_in * flow_in_eff * flow_in_eff_per_distance ** distance
      - where: NOT base_tech=transmission
        expression: flow_in * flow_in_eff
    cost_operation_variable:
      title: Variable operating costs
      description: The operating costs per timestep of a technology.
      default: 0
      unit: cost_per_time
      foreach:
      - nodes
      - techs
      - costs
      - timesteps
      where: cost_export OR cost_flow_in OR cost_flow_out
      equations:
      - expression: timestep_weights * ($cost_export + $cost_flow_out + $cost_flow_in)
      sub_expressions:
        cost_export:
        - where: any(carrier_export, over=carriers) AND any(cost_export, over=carriers)
          expression: sum(cost_export * flow_export, over=carriers)
        - where: NOT (any(carrier_export, over=carriers) AND any(cost_export, over=carriers))
          expression: '0'
        cost_flow_in:
        - where: base_tech=supply
          expression: cost_flow_in * source_use
        - where: NOT base_tech=supply
          expression: sum(cost_flow_in * flow_in, over=carriers)
        cost_flow_out:
        - expression: sum(cost_flow_out * flow_out, over=carriers)
    cost_investment_flow_cap:
      title: Flow capacity investment costs
      description: The investment costs associated with the nominal/rated capacity
        of a technology.
      default: 0
      foreach:
      - nodes
      - techs
      - carriers
      - costs
      where: flow_cap AND (cost_flow_cap OR cost_flow_cap_per_distance)
      equations:
      - expression: $cost_sum * flow_cap
      sub_expressions:
        cost_sum:
        - where: base_tech=transmission
          expression: (cost_flow_cap + cost_flow_cap_per_distance * distance) * 0.5
        - where: NOT base_tech=transmission
          expression: cost_flow_cap
    cost_investment_storage_cap:
      title: Storage capacity investment costs
      description: The investment costs associated with the storage capacity of a
        technology.
      default: 0
      foreach:
      - nodes
      - techs
      - costs
      where: cost_storage_cap AND storage_cap
      equations:
      - expression: cost_storage_cap * storage_cap
    cost_investment_source_cap:
      title: Source flow capacity investment costs
      description: The investment costs associated with the source consumption capacity
        of a technology.
      default: 0
      foreach:
      - nodes
      - techs
      - costs
      where: cost_source_cap AND source_cap
      equations:
      - expression: cost_source_cap * source_cap
    cost_investment_area_use:
      title: Area utilisation investment costs
      description: The investment costs associated with the area used by a technology.
      default: 0
      foreach:
      - nodes
      - techs
      - costs
      where: cost_area_use AND area_use
      equations:
      - expression: cost_area_use * area_use
    cost_investment_purchase:
      title: Binary purchase investment costs
      description: The investment costs associated with the binary purchase of a technology.
      default: 0
      foreach:
      - nodes
      - techs
      - costs
      where: cost_purchase AND purchased_units
      equations:
      - where: base_tech=transmission
        expression: (cost_purchase + cost_purchase_per_distance * distance) * purchased_units
          * 0.5
      - where: NOT base_tech=transmission
        expression: cost_purchase * purchased_units
    cost_investment:
      title: Total investment costs
      description: The installation costs of a technology, including those linked
        to the nameplate capacity, land use, storage size, and binary/integer unit
        purchase.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment_flow_cap OR cost_investment_storage_cap OR cost_investment_source_cap
        OR cost_investment_area_use OR cost_investment_purchase
      equations:
      - expression: sum(default_if_empty(cost_investment_flow_cap, 0), over=carriers)
          + default_if_empty(cost_investment_storage_cap, 0) + default_if_empty(cost_investment_source_cap,
          0) + default_if_empty(cost_investment_area_use, 0) + default_if_empty(cost_investment_purchase,
          0)
    cost_investment_annualised:
      title: Equivalent annual investment costs
      description: An annuity factor has been applied to scale lifetime investment
        costs to annual values that can be directly compared to operation costs. If
        the modeling period is not equal to one full year, this will be scaled accordingly.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment
      equations:
      - expression: $annualisation_weight * $depreciation_rate * cost_investment
      sub_expressions:
        annualisation_weight:
        - expression: sum(timestep_resolution * timestep_weights, over=timesteps)
            / 8760
        depreciation_rate:
        - where: cost_depreciation_rate
          expression: cost_depreciation_rate
        - where: NOT cost_depreciation_rate AND cost_interest_rate=0
          expression: 1 / lifetime
        - where: NOT cost_depreciation_rate AND cost_interest_rate>0
          expression: (cost_interest_rate * ((1 + cost_interest_rate) ** lifetime))
            / (((1 + cost_interest_rate) ** lifetime) - 1)
    cost_operation_fixed:
      title: Total fixed operation costs
      description: The fixed, annual operation costs of a technology, which are calculated
        relative to investment costs. If the modeling period is not equal to one full
        year, this will be scaled accordingly.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment AND (cost_om_annual OR cost_om_annual_investment_fraction)
      equations:
      - expression: "$annualisation_weight * (\n  sum(cost_om_annual * flow_cap, over=carriers)
          +\n  cost_investment * cost_om_annual_investment_fraction\n)"
      sub_expressions:
        annualisation_weight:
        - expression: sum(timestep_resolution * timestep_weights, over=timesteps)
            / 8760
    cost:
      title: Total costs
      description: The total annualised costs of a technology, including installation
        and operation costs.
      default: 0
      unit: cost
      foreach:
      - nodes
      - techs
      - costs
      where: cost_investment_annualised OR cost_operation_variable OR cost_operation_fixed
      equations:
      - expression: default_if_empty(cost_investment_annualised, 0) + $cost_operation_sum
          + default_if_empty(cost_operation_fixed, 0)
      sub_expressions:
        cost_operation_sum:
        - where: cost_operation_variable
          expression: sum(cost_operation_variable, over=timesteps)
        - where: NOT cost_operation_variable
          expression: '0'
      active: true
  constraints:
    flow_capacity_per_storage_capacity_min:
      description: Set the lower bound of storage flow capacity relative to its storage
        capacity.
      foreach:
      - nodes
      - techs
      - carriers
      where: storage_cap AND flow_cap_per_storage_cap_min
      equations:
      - expression: flow_cap >= storage_cap * flow_cap_per_storage_cap_min
    flow_capacity_per_storage_capacity_max:
      description: Set the upper bound of storage flow capacity relative to its storage
        capacity.
      foreach:
      - nodes
      - techs
      - carriers
      where: storage_cap AND flow_cap_per_storage_cap_max
      equations:
      - expression: flow_cap <= storage_cap * flow_cap_per_storage_cap_max
    source_capacity_equals_flow_capacity:
      description: Set a `supply` technology's flow capacity to equal its source capacity.
      foreach:
      - nodes
      - techs
      - carriers
      where: source_cap AND source_cap_equals_flow_cap=True
      equations:
      - expression: source_cap == flow_cap
    force_zero_area_use:
      description: Set a technology's area use to zero if its flow capacity upper
        bound is zero.
      foreach:
      - nodes
      - techs
      where: area_use AND flow_cap_max=0
      equations:
      - expression: area_use == 0
    area_use_per_flow_capacity:
      description: Set a fixed relationship between a technology's flow capacity and
        its area use.
      foreach:
      - nodes
      - techs
      - carriers
      where: area_use AND area_use_per_flow_cap
      equations:
      - expression: area_use == flow_cap * area_use_per_flow_cap
    area_use_capacity_per_loc:
      description: Set an upper bound on the total area that all technologies with
        `area_use` can occupy at a given node.
      foreach:
      - nodes
      where: area_use AND available_area
      equations:
      - expression: sum(area_use, over=techs) <= available_area
    flow_capacity_systemwide_max:
      description: Set an upper bound on flow capacity of a technology across all
        nodes in which the technology exists.
      foreach:
      - techs
      - carriers
      where: flow_cap_max_systemwide
      equations:
      - expression: sum(flow_cap, over=nodes) <= flow_cap_max_systemwide
    flow_capacity_systemwide_min:
      description: Set a lower bound on flow capacity of a technology across all nodes
        in which the technology exists.
      foreach:
      - techs
      - carriers
      where: flow_cap_min_systemwide
      equations:
      - expression: sum(flow_cap, over=nodes) >= flow_cap_min_systemwide
    balance_conversion:
      description: Fix the relationship between a `conversion` technology's outflow
        and consumption.
      foreach:
      - nodes
      - techs
      - timesteps
      where: base_tech=conversion AND NOT include_storage=true
      equations:
      - expression: sum(flow_out_inc_eff, over=carriers) == sum(flow_in_inc_eff, over=carriers)
    flow_out_max:
      description: Set the upper bound of a technology's outflow.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: carrier_out AND NOT operating_units
      equations:
      - expression: flow_out <= flow_cap * timestep_resolution * flow_out_parasitic_eff
    flow_out_min:
      description: Set the lower bound of a technology's outflow.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_out_min_relative AND NOT operating_units
      equations:
      - expression: flow_out >= flow_cap * timestep_resolution * flow_out_min_relative
    flow_in_max:
      description: Set the upper bound of a technology's inflow.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: carrier_in AND NOT operating_units
      equations:
      - expression: flow_in <= flow_cap * timestep_resolution
    source_max:
      description: Set the upper bound of a `supply` technology's source consumption.
      foreach:
      - nodes
      - techs
      - timesteps
      where: source_cap
      equations:
      - expression: source_use <= timestep_resolution * source_cap
    storage_max:
      description: Set the upper bound of the amount of carrier a technology can store.
      foreach:
      - nodes
      - techs
      - timesteps
      where: storage
      equations:
      - expression: storage <= storage_cap
    storage_discharge_depth_limit:
      description: Set the lower bound of the stored carrier a technology must keep
        in reserve at all times.
      foreach:
      - nodes
      - techs
      - timesteps
      where: storage AND storage_discharge_depth
      equations:
      - expression: storage - storage_discharge_depth * storage_cap >= 0
    system_balance:
      description: Set the global carrier balance of the optimisation problem by fixing
        the total production of a given carrier to equal the total consumption of
        that carrier at every node in every timestep.
      foreach:
      - nodes
      - carriers
      - timesteps
      equations:
      - expression: sum(flow_out, over=techs) - sum(flow_in, over=techs) - $flow_export
          + $unmet_demand_and_unused_supply == 0
      sub_expressions:
        flow_export:
        - where: any(carrier_export, over=techs)
          expression: sum(flow_export, over=techs)
        - where: NOT any(carrier_export, over=techs)
          expression: '0'
        unmet_demand_and_unused_supply:
        - where: config.ensure_feasibility=True
          expression: unmet_demand + unused_supply
        - where: NOT config.ensure_feasibility=True
          expression: '0'
    balance_demand:
      description: Set the upper bound on, or a fixed total of, that a demand technology
        must dump to its sink in each timestep.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: base_tech=demand
      equations:
      - where: sink_use_equals
        expression: flow_in_inc_eff == sink_use_equals * $sink_scaler
      - where: NOT sink_use_equals AND sink_use_max
        expression: flow_in_inc_eff <= sink_use_max * $sink_scaler
      sub_expressions:
        sink_scaler:
        - where: sink_unit=per_area
          expression: area_use
        - where: sink_unit=per_cap
          expression: sum(flow_cap, over=carriers)
        - where: sink_unit=absolute
          expression: '1'
    balance_demand_min_use:
      description: Set the lower bound on the quantity of flow a `demand` technology
        must dump to its sink in each timestep.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: sink_use_min AND NOT sink_use_equals AND base_tech=demand
      equations:
      - expression: flow_in_inc_eff >= sink_use_min * $sink_scaler
      sub_expressions:
        sink_scaler:
        - where: sink_unit=per_area
          expression: area_use
        - where: sink_unit=per_cap
          expression: sum(flow_cap, over=carriers)
        - where: sink_unit=absolute
          expression: '1'
    balance_supply_no_storage:
      description: Fix the outflow of a `supply` technology to its consumption of
        the available source.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: base_tech=supply AND NOT include_storage=True
      equations:
      - expression: flow_out_inc_eff == source_use * source_eff
    balance_supply_with_storage:
      description: Fix the outflow of a `supply` technology to its consumption of
        the available source, with a storage buffer to temporally offset the outflow
        from source consumption.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: storage AND base_tech=supply
      equations:
      - expression: storage == $storage_previous_step + source_use * source_eff -
          flow_out_inc_eff
      sub_expressions:
        storage_previous_step:
        - where: timesteps=get_val_at_index(timesteps=0) AND NOT cyclic_storage=True
          expression: storage_initial * storage_cap
        - where: "(\n  (timesteps=get_val_at_index(timesteps=0) AND cyclic_storage=True)\n\
            \  OR NOT timesteps=get_val_at_index(timesteps=0)\n) AND NOT cluster_first_timestep=True"
          expression: (1 - storage_loss) ** roll(timestep_resolution, timesteps=1)
            * roll(storage, timesteps=1)
        - where: cluster_first_timestep=True AND NOT (timesteps=get_val_at_index(timesteps=0)
            AND NOT cyclic_storage=True)
          expression: (1 - storage_loss) ** select_from_lookup_arrays(timestep_resolution,
            timesteps=lookup_cluster_last_timestep) * select_from_lookup_arrays(storage,
            timesteps=lookup_cluster_last_timestep)
    source_availability_supply:
      description: Set the upper bound on, or a fixed total of, a `supply` technology's
        ability to consume its available resource.
      foreach:
      - nodes
      - techs
      - timesteps
      where: source_use AND (source_use_equals OR source_use_max)
      equations:
      - where: source_use_equals
        expression: source_use == source_use_equals * $source_scaler
      - where: NOT source_use_equals AND source_use_max
        expression: source_use <= source_use_max * $source_scaler
      sub_expressions:
        source_scaler:
        - where: source_unit=per_area
          expression: area_use
        - where: source_unit=per_cap
          expression: sum(flow_cap, over=carriers)
        - where: source_unit=absolute
          expression: '1'
    balance_supply_min_use:
      description: Set the lower bound on the quantity of its source a `supply` technology
        must use in each timestep.
      foreach:
      - nodes
      - techs
      - timesteps
      where: source_use_min AND NOT source_use_equals AND base_tech=supply
      equations:
      - expression: source_use >= source_use_min * $source_scaler
      sub_expressions:
        source_scaler:
        - where: source_unit=per_area
          expression: area_use
        - where: source_unit=per_cap
          expression: sum(flow_cap, over=carriers)
        - where: source_unit=absolute
          expression: '1'
    balance_storage:
      description: Fix the quantity of carrier stored in a `storage` technology at
        the end of each timestep based on the net flow of carrier charged and discharged
        and the quantity of carrier stored at the start of the timestep.
      foreach:
      - nodes
      - techs
      - timesteps
      where: (include_storage=true or base_tech=storage) AND NOT (base_tech=supply
        OR base_tech=demand)
      equations:
      - expression: "storage == $storage_previous_step -\n  sum(flow_out_inc_eff,
          over=carriers) + sum(flow_in_inc_eff, over=carriers)"
      sub_expressions:
        storage_previous_step:
        - where: timesteps=get_val_at_index(timesteps=0) AND NOT cyclic_storage=True
          expression: storage_initial * storage_cap
        - where: "(\n  (timesteps=get_val_at_index(timesteps=0) AND cyclic_storage=True)\n\
            \  OR NOT timesteps=get_val_at_index(timesteps=0)\n) AND NOT cluster_first_timestep=True"
          expression: (1 - storage_loss) ** roll(timestep_resolution, timesteps=1)
            * roll(storage, timesteps=1)
        - where: cluster_first_timestep=True AND NOT (timesteps=get_val_at_index(timesteps=0)
            AND NOT cyclic_storage=True)
          expression: (1 - storage_loss) ** select_from_lookup_arrays(timestep_resolution,
            timesteps=lookup_cluster_last_timestep) * select_from_lookup_arrays(storage,
            timesteps=lookup_cluster_last_timestep)
    set_storage_initial:
      description: Fix the relationship between carrier stored in a `storage` technology
        at the start and end of the whole model period.
      foreach:
      - nodes
      - techs
      where: storage AND storage_initial AND cyclic_storage=True
      equations:
      - expression: "storage[timesteps=$final_step] * (\n  (1 - storage_loss) ** timestep_resolution[timesteps=$final_step]\n
          ) == storage_initial * storage_cap"
      slices:
        final_step:
        - expression: get_val_at_index(timesteps=-1)
      active: true
    balance_transmission:
      description: Fix the relationship between between carrier flowing into and out
        of a `transmission` link in each timestep.
      foreach:
      - techs
      - timesteps
      where: base_tech=transmission
      equations:
      - expression: sum(flow_out_inc_eff, over=[nodes, carriers]) == sum(flow_in_inc_eff,
          over=[nodes, carriers])
    symmetric_transmission:
      description: Fix the flow capacity of two `transmission` technologies representing
        the same link in the system.
      foreach:
      - nodes
      - techs
      where: base_tech=transmission
      equations:
      - expression: sum(flow_cap, over=carriers) == link_flow_cap
    export_balance:
      description: Set the lower bound of a technology's outflow to a technology's
        carrier export, for any technologies that can export carriers out of the system.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_export
      equations:
      - expression: flow_out >= flow_export
    flow_export_max:
      description: Set the upper bound of a technology's carrier export, for any technologies
        that can export carriers out of the system.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_export AND export_max
      equations:
      - where: operating_units
        expression: flow_export <= export_max * operating_units
      - where: NOT operating_units
        expression: flow_export <= export_max
    unit_commitment_milp:
      description: Set the upper bound of the number of integer units of technology
        that can exist, for any technology using integer units to define its capacity.
      foreach:
      - nodes
      - techs
      - timesteps
      where: operating_units AND purchased_units
      equations:
      - expression: operating_units <= purchased_units
    available_flow_cap_binary:
      description: Limit flow capacity to zero if the technology is not operating
        in a given timestep.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: available_flow_cap
      equations:
      - expression: available_flow_cap <= flow_cap_max * operating_units
    available_flow_cap_continuous:
      description: Limit flow capacity to the value of the `flow_cap` decision variable
        when the technology is operating in a given timestep.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: available_flow_cap
      equations:
      - expression: available_flow_cap <= flow_cap
    available_flow_cap_max_binary_continuous_switch:
      description: Force flow capacity to equal the value of the `flow_cap` decision
        variable if the technology is operating in a given timestep, zero otherwise.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: available_flow_cap
      equations:
      - expression: available_flow_cap >= flow_cap + ((operating_units - purchased_units)
          * flow_cap_max)
    flow_out_max_milp:
      description: Set the upper bound of a technology's ability to produce carriers,
        for any technology using integer units to define its capacity.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_out AND operating_units AND flow_cap_per_unit
      equations:
      - expression: flow_out <= operating_units * timestep_resolution * flow_cap_per_unit
          * flow_out_parasitic_eff
    flow_in_max_milp:
      description: Set the upper bound of a technology's ability to consume carriers,
        for any technology using integer units to define its capacity.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_in AND operating_units AND flow_cap_per_unit
      equations:
      - expression: flow_in <= operating_units * timestep_resolution * flow_cap_per_unit
    flow_out_min_milp:
      description: Set the lower bound of a technology's ability to produce carriers,
        for any technology using integer units to define its capacity.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_out AND operating_units AND flow_out_min_relative
      equations:
      - where: flow_cap_per_unit
        expression: flow_out >= operating_units * timestep_resolution * flow_cap_per_unit
          * flow_out_min_relative
      - where: available_flow_cap
        expression: flow_out >= available_flow_cap * timestep_resolution * flow_out_min_relative
    storage_capacity_units_milp:
      description: Fix the storage capacity of any technology using integer units
        to define its capacity.
      foreach:
      - nodes
      - techs
      where: storage AND purchased_units AND storage_cap_per_unit
      equations:
      - expression: storage_cap == purchased_units * storage_cap_per_unit
    flow_capacity_units_milp:
      description: Fix the flow capacity of any technology using integer units to
        define its capacity.
      foreach:
      - nodes
      - techs
      - carriers
      where: purchased_units AND flow_cap_per_unit
      equations:
      - expression: flow_cap == purchased_units * flow_cap_per_unit
    flow_capacity_max_purchase_milp:
      description: Set the upper bound on a technology's flow capacity, for any technology
        with integer capacity purchasing.
      foreach:
      - nodes
      - techs
      - carriers
      where: purchased_units
      equations:
      - where: flow_cap_max
        expression: flow_cap <= flow_cap_max * purchased_units
      - where: NOT flow_cap_max
        expression: flow_cap <= bigM * purchased_units
    flow_capacity_minimum:
      description: Set the lower bound on a technology's flow capacity, for any technology
        with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      - carriers
      where: flow_cap_min
      equations:
      - where: NOT purchased_units
        expression: flow_cap >= flow_cap_min
      - where: purchased_units
        expression: flow_cap >= flow_cap_min * purchased_units
    storage_capacity_max_purchase_milp:
      description: Set the upper bound on a technology's storage capacity, for any
        technology with integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: purchased_units AND storage_cap_max
      equations:
      - expression: storage_cap <= storage_cap_max * purchased_units
    storage_capacity_minimum:
      description: Set the lower bound on a technology's storage capacity for any
        technology with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: storage_cap_min
      equations:
      - where: NOT purchased_units
        expression: storage_cap >= storage_cap_min
      - where: purchased_units
        expression: storage_cap >= storage_cap_min * purchased_units
    area_use_minimum:
      description: Set the lower bound on a technology's area use for any technology
        with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: area_use_min
      equations:
      - where: NOT purchased_units
        expression: area_use >= area_use_min
      - where: purchased_units
        expression: area_use >= area_use_min * purchased_units
    source_capacity_minimum:
      description: Set the lower bound on a technology's source capacity for any supply
        technology with a non-zero lower bound, with or without integer capacity purchasing.
      foreach:
      - nodes
      - techs
      where: base_tech=supply AND source_cap_min
      equations:
      - where: NOT purchased_units
        expression: source_cap >= source_cap_min
      - where: purchased_units
        expression: source_cap >= source_cap_min * purchased_units
    unit_capacity_max_systemwide_milp:
      description: Set the upper bound on the total number of units of a technology
        that can be purchased across all nodes where the technology can exist, for
        any technology using integer units to define its capacity.
      foreach:
      - techs
      where: purchased_units AND purchased_units_max_systemwide
      equations:
      - expression: sum(purchased_units, over=nodes) <= purchased_units_max_systemwide
    unit_capacity_min_systemwide_milp:
      description: Set the lower bound on the total number of units of a technology
        that can be purchased across all nodes where the technology can exist, for
        any technology using integer units to define its capacity.
      foreach:
      - techs
      where: purchased_units AND purchased_units_max_systemwide
      equations:
      - expression: sum(purchased_units, over=nodes) >= purchased_units_min_systemwide
    async_flow_in_milp:
      description: Set a technology's ability to have inflow in the same timestep
        that it has outflow, for any technology using the asynchronous flow binary
        switch.
      foreach:
      - nodes
      - techs
      - timesteps
      where: async_flow_switch
      equations:
      - expression: sum(flow_in, over=carriers) <= (1 - async_flow_switch) * bigM
    async_flow_out_milp:
      description: Set a technology's ability to have outflow in the same timestep
        that it has inflow, for any technology using the asynchronous flow binary
        switch.
      foreach:
      - nodes
      - techs
      - timesteps
      where: async_flow_switch
      equations:
      - expression: sum(flow_out, over=carriers) <= async_flow_switch * bigM
    ramping_up:
      description: Set the upper bound on a technology's ability to ramp outflow up
        beyond a certain percentage compared to the previous timestep.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_ramping AND NOT timesteps=get_val_at_index(timesteps=0)
      equations:
      - expression: $flow - roll($flow, timesteps=1) <= flow_ramping * flow_cap
      sub_expressions:
        flow:
        - where: carrier_out AND NOT carrier_in
          expression: flow_out / timestep_resolution
        - where: carrier_in AND NOT carrier_out
          expression: flow_in / timestep_resolution
        - where: carrier_in AND carrier_out
          expression: (flow_out - flow_in) / timestep_resolution
    ramping_down:
      description: Set the upper bound on a technology's ability to ramp outflow down
        beyond a certain percentage compared to the previous timestep.
      foreach:
      - nodes
      - techs
      - carriers
      - timesteps
      where: flow_ramping AND NOT timesteps=get_val_at_index(timesteps=0)
      equations:
      - expression: -1 * flow_ramping * flow_cap <= $flow - roll($flow, timesteps=1)
      sub_expressions:
        flow:
        - where: carrier_out AND NOT carrier_in
          expression: flow_out / timestep_resolution
        - where: carrier_in AND NOT carrier_out
          expression: flow_in / timestep_resolution
        - where: carrier_in AND carrier_out
          expression: (flow_out - flow_in) / timestep_resolution
  piecewise_constraints: {}
  objectives:
    min_cost_optimisation:
      description: Minimise the total cost of installing and operating all technologies
        in the system. If multiple cost classes are present (e.g., monetary and co2
        emissions), the weighted sum of total costs is minimised. Cost class weights
        can be defined in the indexed parameter `objective_cost_weights`.
      equations:
      - where: any(cost, over=[nodes, techs, costs])
        expression: "sum(\n  sum(cost, over=[nodes, techs])\n  * objective_cost_weights,\n\
          \  over=costs\n) + $unmet_demand"
      - where: NOT any(cost, over=[nodes, techs, costs])
        expression: $unmet_demand
      sub_expressions:
        unmet_demand:
        - where: config.ensure_feasibility=True
          expression: "sum(\n  sum(unmet_demand - unused_supply, over=[carriers, nodes])\n\
            \  * timestep_weights,\n  over=timesteps\n) * bigM"
        - where: NOT config.ensure_feasibility=True
          expression: '0'
      sense: minimise
      active: true
BTLF �      ͠             ��5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OCHK    ��                      NAME          nodes +        _Netcdf4Dimid                    Ӄ     �       V                                                                                                                                                                                                                                                                                                                                                         ���BTHD 	      d(i             +g�FSSE      J �    ���GBTHD       d(�     
 
       /@�� ��            �UtgFSSE P     |  �    ���OHDR+                                     *       ��      �       G�     Q            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE #        NAME    	      carriers   �Z��OHDR    �      �                @    *         �    `c     �            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE $        NAME    
      timesteps   Hb�DBTHD       d(�      
 
       �3|�     �ϟ�FRHP               ���������       ��                                                                                  (  b�        0�%BTHD       d(M�              �O�BTHD 	      d(M�              �                  GCOL                        config                applied_math                  defaults
              ��      b               f              ��     �               �               �               �              aluminium_transport_tech�              aluminium_supply�              aluminium_demand�               �              ��     �              ��     �               �               �              Netherlands     �              Iceland �               �              ��     �              ��     �              *�      �               �       	       aluminium       �              ��     �              ��     �              *�      �              a�      �              ��     �              ��     �              *�      �              a�                    ��                  ��                  a�                   ��                  ��                  *�                   a�                   ��                  ��     	             *�      
             a�                                 ��                  ��                  ��                   a�                   ��                  ��                  ��                                 monetary$             ��     %             ��     &             *�      '             a�      )             ��     *             *�      .             *�      /             ��     0             ��      2             *�      3             ��      4             ��      5              6             ��     7              8              9              :             transmission    ;             supply  <             demand  =              ?             ��     @             ��     A              B              C              D              E              F              G      	       aluminium       H              I              J              K      	       aluminium       L              P             ��     Q             ��     R             *�      S              T             ��     U              V              W              X             #8465A9 Y             #F9CF22 Z             #072486 [             ��     \              ]             ��     ^              _              `              a             Aluminium Transport     b             Aluminium Supplyc             Aluminium Demandg             ��     h             ��     i             *�      j             ��     k             default l      	       is_result       m             ��     q             ��     r             ��     s             ��      t             ��     u             ��     y             a�      z             ��     {             ��                  a�      �             ��     �             ��     �             ��     �             ��     �             *�      �             ��     �             a�      �             a�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    OHDR(                                     *       ��            ��     Q           
 ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE          NAME          costs   Y�=          '��OHDR4                                                  ?      @ 4 4�     +         �                   ��                       ��      K�      q�       5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        �>�\FSHD  �                              P x (        F�                    ���xBTLF  _    D     �$ �   1     +�
3 �        �I75 �   7     ��9 �    D     Q�ń T   +     e>� �   2  
   Ŵ�� S   2  	   �]Y� !   2     f�� V   5     bS�� �   l     c���     I      c��� �    o     G��� �   @     K���r                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    D      �    o      V   5      �         �   7      �   @      !   2      S   2  	    �   2  
    �   1      �   l      T   +     �A3�                                                                                                                                                                                                                                                                                                                                FSSE kG     �  n    t�L�FSSE ��     �  P    ����FSSE _$       �    �ukBTHD       d(`_             $�	�FSSE _$     <  �   6)HFSSE ,     4  �    !�1"/�BTHD       d(X             籵BTHD 	      d(X             �q�(FSSE ��      �      �a�!FRHP               ��������      _$                          '                                                         l�      ���           	e\FSSE d	        �      �'    � x    �              �P�FSSE ��      �  M    � bTFHDB ��           ���e     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title    )      Technology flow (a.k.a. nominal) capacity     description    N      A technology's flow capacity, also known as its nominal or nameplate capacity.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �      ��      �      ��      �        _Netcdf4Dimid                                                                                                                                                  FHDB �          Ʊ��     config    �     init:
  name: main model
  calliope_version: 0.7.0
  broadcast_param_data: true
  time_subset:
  - '2005-01-01'
  - '2005-01-31'
  time_resample:
  time_cluster:
  time_format: ISO8601
  distance_unit: km
build:
  mode: plan
  add_math: []
  ignore_mode_math: false
  backend: pyomo
  ensure_feasibility: false
  objective: min_cost_optimisation
  pre_validate_math_strings: true
  operate:
    window: 24h
    horizon: 48h
    use_cap_results: false
solve:
  save_logs:
  solver_io:
  solver_options: {}
  solver: gurobi
  zero_threshold: 1e-10
  shadow_prices: []
  spores:
    scoring_algorithm: integer
    number: 3
    save_per_spore_path:
    skip_baseline_run: false
    tracking_parameter:
    score_threshold_factor: 0.1
 	    def_path    
      model.yaml     serialised_single_element_list  ?      @ 4 4�         serialised_dicts                               ��            ��            ��                                  FHDB �           ɲ� 	    defaults    [     bigM: 1000000000.0
objective_cost_weights: 1
spores_baseline_cost: .inf
spores_slack: 0
spores_score: 0
color: .nan
name: .nan
cap_method: continuous
integer_dispatch: false
include_storage: false
force_async_flow: false
flow_cap_per_storage_cap_min: 0
flow_cap_per_storage_cap_max: .inf
flow_cap: .inf
flow_cap_max: .inf
flow_cap_max_systemwide: .inf
flow_cap_min: 0
flow_cap_min_systemwide: 0
flow_out_min_relative: 0
flow_cap_per_unit: .nan
flow_in_eff: 1.0
flow_out_eff: 1.0
flow_out_parasitic_eff: 1.0
flow_ramping: 1.0
export_max: .inf
lifetime: .inf
area_use: .inf
area_use_max: .inf
area_use_min: 0
area_use_per_flow_cap: .nan
storage_cap: .inf
storage_cap_max: .inf
storage_cap_min: 0
storage_cap_per_unit: .nan
storage_discharge_depth: 0
storage_initial: 0
storage_loss: 0
cyclic_storage: true
purchased_units_min_systemwide: 0
purchased_units_max_systemwide: .inf
purchased_units: .inf
purchased_units_min: 0
purchased_units_max: .inf
sink_unit: absolute
sink_use_min: 0
sink_use_max: .inf
sink_use_equals: .nan
source_unit: absolute
source_cap_equals_flow_cap: false
source_eff: 1.0
source_use_min: 0
source_use_max: .inf
source_use_equals: .nan
source_cap: .inf
source_cap_max: .inf
source_cap_min: 0
one_way: false
distance: 1.0
flow_in_eff_per_distance: 1.0
flow_out_eff_per_distance: 1.0
cost_flow_cap_per_distance: 0
cost_purchase_per_distance: 0
cost_flow_cap: 0
cost_export: 0
cost_interest_rate: 0
cost_om_annual: 0
cost_om_annual_investment_fraction: 0
cost_flow_in: 0
cost_flow_out: 0
cost_purchase: 0
cost_area_use: 0
cost_source_cap: 0
cost_storage_cap: 0
cost_depreciation_rate: 1
available_area: .inf
     allow_operate_mode                                name    
      main model     timestamp_model_creation  ?      @ 4 4�                ��E�A     timestamp_build_start  ?      @ 4 4�                ��F�A     timestamp_build_complete  ?      @ 4 4�                R�F�A                          FHDB �           ]     termination_condition          optimal     calliope_version_defined          0.7.0     calliope_version_initialised    
      0.7.0.dev6     applied_overrides            	    scenario          None     timestamp_solve_start  ?      @ 4 4�                ��F�A     timestamp_solve_complete  ?      @ 4 4�                �'�F�A     serialised_bools  ?      @ 4 4�         serialised_nones          scenario     serialised_sets  ?      @ 4 4�         _NCProperties    "      version=2,netcdf=4.9.2,hdf5=1.14.4                                                                                                                                                                                                                                                                                                                                                                                                        FHIB �           b�      b�      ������������������������������������������������b�      �������������������������o�TREE  ����������������                                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�                      ?      @ 4 4�     +         �                   ��                 ��      ~�      ��       #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      	4�FRHP               ���������       ��                                                                                  (  ��        qp+�BTHD       d(�              u�`�BTHD 	      d(�              މ�rFSHD  �                              P x (        G�                    �P�~BTLF  W    D     �$ �   1     +�
3 y        �I75 �   7     ��9 �    -     Q�ń �   +     e>� s   2  
   Ŵ�� A   2  	   �]Y�    2     f�� D   5     bS��    L     c���     A      c��� �    |     G��� �   @     K���Җ�]                                                                                                                                                                                                                                                                        BTLF 	     A       W    D      �    -      �    |      D   5      y         �   7      �   @         2      A   2  	    s   2  
    �   1      �   +         L     o �                                                                                                                                                                                                                                                                                                                                FRHP               ��������|       P                                                                                 (  �d       �}           2�B�x^c`�



~,_����z�z M�KFHDB ��           ����     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      �     title          Link flow capacity     description    [      A transmission technology's flow capacity, also known as its nominal or nameplate capacity.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         _Netcdf4Dimid                     DIMENSION_LIST                              ��      f                                                                                                                                                                                      TREE  ����������������                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR]D                         �                           �          ?      @ 4 4�     +         �                   �              �          t
     ��       �       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               h��wFRHP               ��������*                                                                                        (  �       1'�FSHD  *                              P x (        �y                   ��GDBTLF  c    D     �$ �   1     +�
3 �         �I75 �   7     ��9 �    *     Q�ń �   +     e>� �   2  
   Ŵ�� �   2  	   �]Y� h   2     f�� �   5     bS�� /   |     c���     M      c��� �    �     G��� (   @     K���>���                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    *      �    �      �   5      �          �   7      (   @      h   2      �   2  	    �   2  
    �   1      /   |      �   +     x��                                                                                                                                                                                                                                                                                                                                BTHD       d(�u             ���BTHD 	      d(�w             *���FSHD  �                              P x (        nF                   �e�BTHD 	      d(��             )�         �d^BTHD 	      d(`a             ���FSHD  |                              P x (        �                    2ϖ\      !o��x^c`@?~,��w  'FFHDB t
          �C�     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier outflow     description    �      The outflow of a technology per timestep, also known as the flow discharged (from `storage` technologies) or the flow received (by `transmission` technologies) on a link.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �      ��      �      ��      �      ��      �        _Netcdf4Dimid                                                           TREE  ����������������a                                               a0                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR D                         �                           �          ?      @ 4 4�     +         �                   �6              �          1+     �+     �+                                         ��GoBTHD       d(�$     ' '       x��XBTHD      d(�I      '       W_`FSHD                               P x          El     ,       ,       $F��BTLF i�8 �   ��� �   �J 1   y}W !   ���  �  # �rQ' �    5�. �   �8 �    ���I �  ) )m�M �  & y��P |    `��P    ��-S B  , w�iV *   ���X    ���[ >  # �^P^ �   ��l �     ��{�    q��� �   1M7� �  " D9� }   �ԕ� �   ��_� `    J鱷 n  ' 7���   - ĕ�� �    ho�     ��� I   �L�� �   S�:� �   Ѧ� -    ��� �  * d�� E    �;�� a  " yܴ� P  ! �}"� �   XX� q  $ ��� `   �]�                                                                         BTLF         �            �             �            �    	           
        !           >  #         a  "         �  *         �           -                        E            `            |            �  " E㸑                                                                                                                                                                                                                                                                      OHDR.D                         �                           �         +         �                   Y�              �          �     �)     �         ?      @ 4 4�   M                                                                                 q��BTHD       d(��             +}9%r��<OHDR 4                  �                    �         +         �                   P           �          �F     G     EG         ?      @ 4 4�   3                                                       ��                FRHP               ��������4       ,                                                                                 (  �0       �4e�BTHD       d(a,             ���#BTHD 	      d(a.             �?xFSHD  4                              P x (        ��                    6g�BTLF  c    D     �$ �   1     +�
3 �         �I75 �   7     ��9 �    )     Q�ń �   +     e>� �   2  
   Ŵ�� �   2  	   �]Y� ^   2     f�� �   5     bS�� %   |     c���     M      c��� �    �     G���    @     K���	;Lb                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    )      �    �      �   5      �          �   7         @      ^   2      �   2  	    �   2  
    �   1      %   |      �   +     8�                                                                                                                                                                                                                                                                                                                                x^��1�0AC3�0�Pt��x4�.�+n-                      nx`�� ������ �� T��0���{ ���c���O�yFHDB 1+          ��     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier inflow     description    �      The inflow to a technology per timestep, also known as the flow consumed (by `storage` technologies) or the flow sent (by `transmission` technologies) on a link.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �      ��      �      ��      �      ��      �        _Netcdf4Dimid                                                                     FHDB a�          �+�       nodes��            techs��            carriers*�             	timestepsa�             costs��             flow_cap��             link_flow_cap�             flow_out            flow_in#     	       
source_use*     
       
source_capQ\            flow_out_inc_eff�r            flow_in_inc_eff�(            cost_operation_variable�y            cost��            capacity_factor��            bigMA1        TREE  ����������������T                                               �K                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     FRHP               ���������      ��                           
                                                      (  	!       �)��BTHD 	      d(�     
 
       ��ə      �(            k��NFRHP               ���������       �                                                                                 (  �{       �o3FSSE �     �  R    eU8    FRHP               ���������       kG                                                                                 (  L       ��BTHD       d(�G             �Ƀ�BTHD 	      d(�I             e�FSHD  �                              P x (        M�                    (��BTLF  _    D     �$ �   1     +�
3 y         �I75 �   7     ��9 �    *     Q�ń C   +     e>� t   2  
   Ŵ�� B   2  	   �]Y�    2     f�� D   5     bS�� �   l     c���     I      c��� �    w     G��� �   @     K����X                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    *      �    w      D   5      y          �   7      �   @         2      B   2  	    t   2  
    �   1      �   l      C   +     O��                                                                                                                                                                                                                                                                                                                                x^��1  0�!	��"#{g��%"2                      ���w z� D8 D���� � B���H}�yFHDB �F          {��     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title          Source flow use     description    V      The carrier flow consumed from outside the system boundaries by a `supply` technology.     default                                 unit          energy 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��            ��           ��             _Netcdf4Dimid                                                                                                                                                                   TREE  ����������������S                                       Bd                           �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             OHDR$                                    ?      @ 4 4�     +         �                   �h                   �      ��      *      J                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             �"�bBTLF  [    D     �$ �   1     +�
3 �        �I75 �   7     ��9 �    /     Q�ń Y   +     e>� �   2  
   Ŵ�� h   2  	   �]Y� 6   2     f�� k   5     bS�� �   \     c���     E      c��� �    �     G��� �   @     K����闯                                                                                                                                                                                                                                                                        BTLF 	     E       [    D      �    /      �    �      k   5      �         �   7      �   @      6   2      h   2  	    �   2  
    �   1      �   \      Y   +     ���                                                                                                                                                                                                                                                                                                                                OCHK   k     2      +        _Netcdf4Dimid                :        units          hours since 2005-01-01 00:00:001    	    calendar          proleptic_gregorian  Х�      �y            ���[x^��1  0�!	��"'������                     ���  @��� �>LP�  }g�/�FHDB �           0�     _Netcdf4Coordinates                                    _FillValue  ?      @ 4 4�                      �     title          Source flow capacity     description    |      The upper limit on a flow that can be consumed from outside the system boundaries by a `supply` technology in each timestep.     default                                 unit          power 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �      ��      �        _Netcdf4Dimid                                                                                                                                             TREE  ����������������                               �{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDRD                         �                           �          ?      @ 4 4�     +         �                   �              �          �E     X     ~      *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ��L�BTLF  c    D     �$ z   1  
   +�
3 m   7     ��9 �    ;     Q�ń '   +     e>� H   2  	   Ŵ��    2     �]Y� �   2     f�� 8   5     bS�� �   |     c���     M      c��� �    V     G��� �   @     K���!���                                                                                                                                                                                                                                                                                         BTLF 	     M       c    D      �    ;      �    V      8   5      m   7      �   @      �   2         2      H   2  	    z   1  
    �   |      '   +     5���                                                                                                                                                                                                                                                                                                                                             FSSE      *  �    ���(OHDR�D                         �                           �         +         �                   ��              �          �     ��     ʩ         ?      @ 4 4�   �                                                                                                                                                                                                    �}�       8M=x^c`�
@ď�?��ޡ D��FHDB �E          ����     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title           Carrier outflow including losses     description    5      Outflows after taking efficiency losses into account.     default                             
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��           ��             _Netcdf4Dimid                                                                                                                                                                                               TREE  ����������������a                                               ��                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     FSSE �     �  m    �� �FSSE h�     �  i    Mm(�FSSE 5#     � M    �f��   /+NfFSSE �a     ~ �    ӹ��BTHD       d( w     	 	       ���8BTHD 	      d( y     	 	       	lx8FSSE �8     � 	    ��KPFSSE      �     �qG��BTHD       d(.b             �U��      xuk    �	�OCHK   5m     B      +        _Netcdf4Dimid                  T��oBTHD 	      d(�     
 
       �"�rE �     �  B    {��mBTHD       d(��             <5�#          ��j         FRHP               ���������       ��                                                                                 (  Y�       �J\FSHD  �                              P x (        h�                    9��BTLF  c    D     �$ x   1  
   +�
3 k   7     ��9 �    :     Q�ń %   +     e>� F   2  	   Ŵ��    2     �]Y� �   2     f�� 6   5     bS�� �   |     c���     M      c��� �    U     G��� �   @     K�����o�                                                                                                                                                                                                                                                                                         BTLF 	     M       c    D      �    :      �    U      6   5      k   7      �   @      �   2         2      F   2  	    x   1  
    �   |      %   +     �5��                                                                                                                                                                                                                                                                                                                                             x^��1�0AC3�0�Pt��x4�.�+n-                      nx`�� ������ �� T��0���{ ���c���O�yFHDB �          Ӷ$�     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Carrier inflow including losses     description    4      Inflows after taking efficiency losses into account.     default                             
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��      	     ��      
       _Netcdf4Dimid                                                                                                                                                                                                 TREE  ����������������T                                               B�                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OCHK   �     �       +        _Netcdf4Dimid                  鈞yBTHD 	      d(�5     
 
        �z*BTHD       d( K     
 
       %s� k�a%OCHK   �     2      +        _Netcdf4Dimid                  ^�_           ��BTHD 	      d(��             ��(OCHK    C�     Q                NAME          techs   Fr/�   FRHP               ���������       �                                                                                 (  ��       ���BTHD       d(B�             -��BTHD 	      d(B�              ZE�FSHD  �                              P x (        �                   %�(BTLF  c    D     �$ �   1     +�
3 a   '     �I75 �   7     ��9 �    3     Q�ń B   +     e>� c   2  
   Ŵ�� 1   2  	   �]Y� �   2     f�� ,   5     bS�� �   |     c���     M      c��� �    R     G��� �   @     K���/���                                                                                                                                                                                                                                                                        BTLF 	     M       c    D      �    3      �    R      ,   5      a   '      �   7      �   @      �   2      1   2  	    c   2  
    �   1      �   |      B   +     ]Ě                                                                                                                                                                                                                                                                                                                                x^��1  0�!	��"#{g��%"2                      ���w z� D8 D���� � B���H}�yFHDB �          `�~     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      �     title          Variable operating costs     description    1      The operating costs per timestep of a technology.     default                                 unit          cost_per_time 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��           ��             _Netcdf4Dimid                                                                                                                                                                    TREE  ����������������a                                               ��                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�4                                                  ?      @ 4 4�     +         �                   O�                      ��     ֏     ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ��|uFRHP               ���������       h�                                                                                 (  O�       �_FSHD  �                              P x (        2�                   bg.BTLF  _    D     �$ �   1     +�
3 v        �I75 �   7     ��9 �    &     Q�ń >   +     e>� o   2  
   Ŵ�� =   2  	   �]Y�    2     f�� A   5     bS�� �   l     c���     I      c��� �    x     G��� �   @     K���k�                                                                                                                                                                                                                                                                        BTLF 	     I       _    D      �    &      �    x      A   5      v         �   7      �   @         2      =   2  	    o   2  
    �   1      �   l      >   +     su�                                                                                                                                                                                                                                                                                                                                OHDR                                     *       ��      �       ��      �           ������������������������A         _Netcdf4Coordinates                            +        CLASS          DIMENSION_SCALE   �%�lx��WOHDR                                     *       ��      b       ɨ     F            ������������������������A         _Netcdf4Coordinates                           +        CLASS          DIMENSION_SCALE   �2���            ��q�x^��1�0�*A�pP9�@VG���~/ѐ1                       �]���iA���p��� �4!��{8��{ z���r�FHDB ��          ^9�     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      �     title          Total costs     description    W      The total annualised costs of a technology, including installation and operation costs.     default                                 unit          cost 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��           ��             _Netcdf4Dimid                                                                                                                                                                        TREE  ����������������                                        ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         x^c`�	��;Ï�?��w�?X_ M�	   ��      �      ��      �      ��      �      ��      �      ��      �   	   ��      �      ��        x^��1 0�@�տ�J��
(�iȐ*                          �υ�{ ���c`��&:�@v �U~L� ��x^c`� 0�� 	Ox^c�f`a`X�����𐁡��Gʏz�z 7�qx^{�C����� �2    ��.Ax^c` �� '0   ��      <     ��      ;     ��      :      ��      L  	   ��      K      ��      J  	   ��      G      ��      H      ��      I  x^c` ~|���Ǉz���z{ ?�;   ��      Z     ��      Y     ��      X  x^c`@?>,��w  �>   ��      c     ��      b     ��      a  x^c` ~����Ǐ�z{��z ?{;x^c`@?~|��� :x^c`@?~|��� :x^c�f 0�L��ɇ`�
D�H���Ǐz����� ��x^{i��h��Ѳ��C?��� (y`x^[�p�����d�3O�
p  G-x^��1   �)��؃�d���h                 ��m۶m۶G��m۶m۞��V5۶m۶m���;�m۶m۶G~ˁ{x^��1   �)��؃�d���h                 _�m۶m۶G��m۶m۞�V5۶m۶m�;�m۶m۶�~��{x^c`ddd`   x^c`x��`���a!CÔ?���;  I��x^��1  �5�#X�o@   ���JOp�x^��1  �5�#X�o@   ���JOp�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       FSHD  �                             P x (        !�                   �m�FSSE ;      t �    @�[YFHIB a�          �4     	%     �?     �������� "BTHD       d(�3     
 
       �P�      �y            o*�                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      FSSE ��     � d    �o��FSSE �3     � x    diqBOCHK    �I     �       /        serialised_bools    	      is_result  $,b�   ��            y��&OHDR�D                         �                           �          ?      @ 4 4�     +         �                   �
              �         
 ��     ��      ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            �<9FRHP               ��������t      ;                            
                                                      (  �       F1��FSHD  t                             P x (        Q�                   �@��BTLF  c    D     �$ �   1     +�
3 �    7     ��9 a   +  	   e>� �   2     Ŵ�� P   2     �]Y�    2     f�� �   |     c���     M      c��� �    @     K������                                                                                                                                                                                                                                                                                                                                            BTLF 	     M       c    D      �    7      �    @         2      P   2      �   2      �   1      �   |      a   +  	   #��                                                                                                                                                                                                                                                                                                                                                                                     FRHP               ���������                                 	                                                      (  �       �)��FSHD  �                             P x (        �                   �iBTHD       d(��     
 
       �ROHDR4                                                 +         �                   E=                     
 3     ��     �         ?      @ 4 4�   I                                                                             \m��           �X�MFHDB ��          �|,S     _Netcdf4Coordinates                                          _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      $     ��      %     ��      &     ��      '       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                     TREE  ����������������T                                               �                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     OHDR�$                                    ?      @ 4 4�     +         �                   	'                  
 
E     Բ      �E      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ���BTLF  [    D     �$ �   1     +�
3 �    7     ��9 9   +  	   e>� z   2     Ŵ�� H   2     �]Y�    2     f�� �   \     c���     E      c��� �    @     K���ۿ��                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7      �    @         2      H   2      z   2      �   1      �   \      9   +  	   �2k|                                                                                                                                                                                                                                                                                                                                                                                    OHDR2$                                   +         �                   2T                  
 J     �     �J         ?      @ 4 4�   y                                                                                                                             �5�k)<�FHDB 
E          ��=&     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      )     ��      *       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                            FHDB a�         �C�Y       systemwide_capacity_factor            systemwide_levelised_cost�            total_levelised_cost�            objective_cost_weights�^            	base_tech0s            carrier {            carrier_out�u            color O            flow_cap_max�            name�}             
carrier_in��     !       flow_out_eff��     "       one_way^�     #       cost_flow_in��     $       latitude��                     TREE  ����������������                               s�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�           ?      @ 4 4�     *         �    W�     U            ������������������������D         _FillValue  ?      @ 4 4�                      �7    
    is_result                            A        default  ?      @ 4 4�                    e��A@        serialised_single_element_list  ?      @ 4 4�    2        serialised_dicts  ?      @ 4 4�      �p��    FRHP               ���������      �3                           
                                                      (  E9       r׆�FSHD  �                             P x (        <�                   ϟ�8BTLF  _    D     �$ �   1     +�
3 �    7     ��9 M   +  	   e>� ~   2     Ŵ�� L   2     �]Y�    2     f�� �   l     c���     I      c��� �    @     K�����1�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   2      ~   2      �   1      �   l      M   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    FRHP               ���������      �8                           	                                                      (  ^�       ���FSHD  �                             P x (        ׎                   .ҝ�BTHD       d(^�     	 	       ;�pBTHD 	      d(^�     	 	       ���BFSSE ��     � u    �>P   ;�}OFHDB 3          ���     _Netcdf4Coordinates                                      _FillValue  ?      @ 4 4�                      � 
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      .     ��      /     ��      0       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                        TREE  ����������������                                        ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         OCHK    ��            2        serialised_nones  ?      @ 4 4�    1        serialised_sets  ?      @ 4 4�    5 ��  FRHP               ���������      �J                           
                                                      (  2P       Ȯ�vBTHD 	      d( M     
 
       �Ƌ,FSHD  �                             P x (        j^                   Q�9BTLF  [    D     �$ �   1     +�
3 �    7     ��9 9   +  	   e>� z   2     Ŵ�� H   2     �]Y� �    2     f�� �   \     c���     E      c���    @     K����2��                                                                                                                                                                                                                                                                                                                                            BTLF 	     E       [    D      �    7         @      �    2      H   2      z   2      �   1      �   \      9   +  	   �|<�                                                                                                                                                                                                                                                                                                                                                                                    OHDR                                      *       ��      S      
 �     y�     ��         q�     0        p                                                                                                                    �-Yf  b��,FHDB J          }	�     _Netcdf4Coordinates                                   _FillValue  ?      @ 4 4�                      �     serialised_dicts  ?      @ 4 4�     
    is_result                                serialised_single_element_list  ?      @ 4 4�         serialised_bools  ?      @ 4 4�         serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      2     ��      3       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                               ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          FSSE �J     � d    Y��jOHDR�                      ?      @ 4 4�     +         �                    k                $a     �     �a      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 CcssFRHP               ��������~      �a                                                                                 (   g       C��BTHD 	      d(.d             0E<7FSHD  ~                             P x (        p�                   ^i��BTLF  W    D     �$ �   1     +�
3 �    7     ��9 W   +  
   e>� �   2     Ŵ�� y   /     �]Y� G   2     f�� �    5     bS��    L  	   c���     A      c���    @     K�����                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    5         @      G   2      y   /      �   2      �   1         L  	    W   +  
   &��                                                                                                                                                                                                                                                                                                                                                                       FRHP               ���������      ��                           
                                                      (  �;       ^��BTHD       d(R6     
 
       �i      �^             x%GrFHDB $a          �Z��     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      4       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                              TREE  ����������������                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�                                     *       ��      5      ��     0           	 �7     ��     ��      �                                                                    7                                                          @                                                                   2                                                     /                                                  2                                                     1                                                    L                                                                               R�QOHDRO4                                                 +         �                   �                     
 �     q     ~�         ?      @ 4 4�   �                                                                                                                                      �Y�t   �-�BTLF  a   1     +�
3 W    7     ��9 �   +     e>� /   2     Ŵ��     /     �]Y� �    2     f�� �   L     c���     A      c��� �    @     K����;��                                                                                                                                                                                                                                                                                                                                                             BTLF 	     A       W    7      �    @      �    2          /      /   2      a   1      �   L      �   +     ��|                                                                                                                                                                                                                                                                                                                                                                                                 OHDR�$                                                   *       ��      =      ��     `           	 �     �8     �8                                                                              7                                                          @                                                                   2                                                     /                                                  2                                                     1                                                    \                                                                                               ��&�OHDRU                                     *       ��      \      
 ��     3     ��         ��     0        �                                                                                                                                                                                                         |��VBTHD       d(��     
 
       %U�       ���FRHP               ��������r      �                                                                                 (  d�       ���dFSHD  r                             P x (        X�                   8��+   {            ��q�FHDB �7          m��     _Netcdf4Coordinates                            
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      6       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       BTLF  e   1     +�
3 [    7     ��9 �   +     e>� 3   2     Ŵ��    /     �]Y� �    2     f�� �   \     c���     E      c��� �    @     K���e���                                                                                                                                                                                                                                                                                                                                                             BTLF 	     E       [    7      �    @      �    2         /      3   2      e   1      �   \      �   +     3it.                                                                                                                                                                                                                                                                                                                                                                                                 OHDR                       ?      @ 4 4�     +         �                    �     ��     I�         �            [                                                                                               I(�FSSE ��     r �    1�.b��BTHD       d(°             ��F�BTHD 	      d(²             Q}c                        FRHP               ���������      ��                           
                                                      (  �       �]$�BTHD 	      d(��     
 
       ]��FSHD  �                             P x (        #9                   "t�)BTLF  _    D     �$ �   1     +�
3 �    7     ��9 J   +  	   e>� {   2     Ŵ�� L   /     �]Y�    2     f�� �   l     c���     I      c��� �    @     K���JC~�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   /      {   2      �   1      �   l      J   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    OHDR                       ?      @ 4 4�     +         �                    ��     c�     ��         ��            [                                                                                               $B�N  �u            ~���FHDB �               _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      P     ��      Q     ��      R       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                            FHDB �          W�r�     _Netcdf4Coordinates                                
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      ?     ��      @       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                                       Y�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FSSE �     r �    a]hFSSE ť     � J    d�z�BTHD 	      d(��     
 
       t*ԭBTHD       d(��     
 
       ���             FRHP               ���������      ť                           
                                                      (  �       3�w8BTHD       d(�     
 
       $���BTHD 	      d(�     
 
       ���FSHD  �                             P x (        s�                   �z��BTLF  �   1     +�
3 W    7     ��9    +  	   e>� p   2     Ŵ�� A   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c��� �    @     K����z.                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    7      �    A      �    @         2      A   /      p   2      �   1      �   L         +  	   �`�E                                                                                                                                                                                                                                                                                                                                                                                    FHDB �          rT"     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      T       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDR�                      ?      @ 4 4�     +         �                   d�                d     ��     ��      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             [=BTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       BTHD       d(�P             Y���  ����OHDR4    �                    �                       +         �                   �Z     �                �O     ´     YP         ?      @ 4 4�   I                                                                             �Ց        ��v�BTHD 	      d(R8     
 
       v漚FSSE ��     � M    �dB�   ���FHDB d          EQ�G     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      [       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR4                                                 +         �                   ��                     
 ��     ��     W�         ?      @ 4 4�   I                                                                             �#��FSSE h�     � J    Q�2P     FRHP               ���������      h�                           
                                                      (  l�       0L��FSHD  �                             P x (        ��                   ?ݮ;BTLF  �   1     +�
3 W    7     ��9    +  	   e>� p   2     Ŵ�� A   /     �]Y�    2     f�� �    A     bS�� �   L     c���     A      c��� �    @     K����z.                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    7      �    A      �    @         2      A   /      p   2      �   1      �   L         +  	   �`�E                                                                                                                                                                                                                                                                                                                                                                                    OHDR                       ?      @ 4 4�     +         �                    �     ��     )V         �            [                                                                                               ���BTHD       d(�             {��FSHD  �                             P x (        B�                   "�\w       BTHD       d(�#     
 
       Y~FHDB ��          Ĝz     _Netcdf4Coordinates                            
    is_result                                 default  ?      @ 4 4�                      ��     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      ]       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                      FSSE }�     � u    }��2BTHD       d(��             {ae                        FRHP               ���������      }�                           
                                                      (  ��       ����BTHD 	      d(��     
 
       QêFSHD  �                             P x (        l�                   me�BTLF  _    D     �$ �   1     +�
3 �    7     ��9 J   +  	   e>� {   2     Ŵ�� L   /     �]Y�    2     f�� �   l     c���     I      c��� �    @     K���JC~�                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    D      �    7      �    @         2      L   /      {   2      �   1      �   l      J   +  	   ���                                                                                                                                                                                                                                                                                                                                                                                    OHDR`4                                                 +         �                   c                      �     ��     �         ?      @ 4 4�   �                                                                                                                                                       �G�7  �u        BTHD 	      d(��             p��FSSE ��     K �    S�#   �ӸFHDB ��          ��     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      g     ��      h     ��      i       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                            TREE  ����������������                                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FRHP               ��������r      ��                                                                                 (  ��       �|��BTHD       d(�             )��BTHD 	      d(�             n��FSHD  r                             P x (        j�                   �jBTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OCHK    �            �  	   0   REFERENCE_LIST 6     dataset        dimension                         �y            ��            �            �            �^             ��            �U��BTHD 	      d(�%     
 
       B��z     FRHP               ��������V                                                                                       (  c       @e�BTHD       d(q             U�L�         !}�8FHDB ��          )     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      j       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FRHP               ��������K      ��                                                                                 (  ��       ����FSHD  K                             P x (        o�                   u=r[BTLF  W    D     �$    1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� y   b     �]Y� G   2     f�� �    5     bS�� >   L  	   c���     A      c���    @     K�������                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    5         @      G   2      y   b      �   2         1      >   L  	    �   +  
   ���                                                                                                                                                                                                                                                                                                                                                                       OHDR�                      ?      @ 4 4�     +         �                   
 �"     E�     ��         �+                                                                                                                                                                                                                                                                                            �%AFSSE      V �    ���$    GL��FHDB �          k�f     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools                            	   ��      l     ��      k       serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      m       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                           TREE  ����������������                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTHD 	      d(q             (���FSHD  V                             P x (        ��                   `7�BTLF  _    D     �$ �   1     +�
3 �    7     ��9    +  
   e>� �   2     Ŵ�� �   /     �]Y� O   2     f�� �    5     bS��    l  	   c���     I      c���    @     K�����!�                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    5         @      O   2      �   /      �   2      �   1         l  	       +  
   {�K�                                                                                                                                                                                                                                                                                                                                                                       OHDR 4                                                            +   �                   m�                     
 �~     �N     ]V      B                                                                      J$-� FRHP               ��������J                                                                                       (  �n       F�+�FSHD  J                             P x (        ��                    �B�FHDB �          �!�V     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      q     ��      r     ��      s       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                       TREE  ����������������&                                        �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FRHP               ���������      5#                           
                                                      (  �'       ��t�FSHD  �                             P x (        M�                   �k�=BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    FHDB �"          ���M     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      t       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                    TREE  ����������������                       F�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�                      ?      @ 4 4�     +         �                   �A               
 .f     �f     �      �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             y�BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    OHDRO4    �                    �                       +         �                   �r     �                ~     �O     ��          ?      @ 4 4�   �                                                                                                                                      O@Z           �E��FHDB .f          �C     _Netcdf4Coordinates                                 _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      u       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                    FHDB a�         ��&�%       	longitude�3     &       sink_use_max�     '       source_use_maxR:     (       definition_matrixq     )       distance��     *       timestep_resolution�T     +       timestep_weights�M                                                                                                                                                                                                                                                                            TREE  ����������������                       ^�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTIN           - �&     �K     mjH                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     BTLF         B  ,         n  '         �           �  )         �           �           �                      *           I            `   !        }   "        �   #        �   $        �   %           &        1   '        P  ! (        q  $ )        �   *        �  & +        �  # ךm�                                                                                                                                                                            OHDR     �      �          ?      @ 4 4�     +         �                   
 ��     EO     ��         �     �      [                                                                                               '�  ��rIBTHD       d(�     
 
       �Z�|  _y�LFSSE P     J �    ���ABTHD       d(�     
 
       �<V�  ׷�FSSE ��     r �    ����                     BTHD       d(g             .vݎFRHP               ��������J      P                                                                                 (  �V       *ZyjBTHD 	      d(�R             9�ߔFSHD  J                             P x (        *O                   	�2�BTLF  _    D     �$ �   1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� �   /     �]Y� [   2     f�� �    A     bS��    l  	   c���     I      c���    @     K������g                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    A         @      [   2      �   /      �   2      �   1         l  	    �   +  
   �0#q                                                                                                                                                                                                                                                                                                                                                                       OHDR0    �      �          ?      @ 4 4�     +         �                   
 �l     �V     wn         L�     �      �                                                                                                                                               ����FSSE ��     � M    �\� BTHD 	      d(�             ��O� O           BTHD 	      d(Ӂ     
 
       2�u0FSSE �     � P    �d&BTHD       d(L�     
 
       WvC          ��e�FHDB �O          �S�7     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      y     ��      z     ��      {       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                          TREE  ����������������W                                       w�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  _    D     �$ �   1     +�
3 �    7     ��9 �   +  
   e>� �   2     Ŵ�� �   /     �]Y� [   2     f�� �    A     bS��    l  	   c���     I      c���    @     K������g                                                                                                                                                                                                                                                                                                                           BTLF 	     I       _    D      �    7      �    A         @      [   2      �   /      �   2      �   1         l  	    �   +  
   �0#q                                                                                                                                                                                                                                                                                                                                                                       OCHK    ��     @          0   REFERENCE_LIST 6     dataset        dimension                                     #            *            �r            �(            �y            ��            �             R:             �T             �M             �N1FSSE _$       �     �    �   �]FSSE ��     � M    �<�_      OFRHP               ���������      ��                           
                                                      (  L�       �q1�           b��OCHK    ��                0   REFERENCE_LIST 6     dataset        dimension                         ��                         #            �r            �(            ��                        �             �             �u            ��            q            ��eBTHD 	      d(L�     
 
       �  ���       �~�FHDB ~          ���     _Netcdf4Coordinates                                       _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��           ��      �     ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                          TREE  ����������������U                                       ��             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           FRHP               ���������      �                           
                                                      (  m�       ���_FSHD  �                             P x (        �V                   �^��BTLF  �        I�B i   1     +�
3 _    7     ��9 %   +  	   e>� 7   2     Ŵ��    /     �]Y� �    2     f�� �   l     c���     I      c��� �    @     K�����ߘ                                                                                                                                                                                                                                                                                                                                            BTLF 	     I       _    7      �    @      �    2         /      7   2      i   1      �         �   l      %   +  	   �~$                                                                                                                                                                                                                                                                                                                                                                                    OCHK�    0   REFERENCE_LIST 6     dataset        dimension                         ��                           #             *             Q\             �r             �(             �y             ��             ��              {             �u             ��             ��             ��             �3             �            R:            q             
V8^FHDB �~          ���     _Netcdf4Coordinates                                   
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         dtype          bool     DIMENSION_LIST                              ��      �     ��      �     ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                 TREE  ����������������                                       #�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         FSHD  �                             P x (        V                   tܚ       FRHP               ��������r      ��                                                                                 (  �       ����FSHD  r                             P x (        qO                   �� _BTLF  W    D     �$ �   1     +�
3 �    7     ��9 c   +  
   e>� �   2     Ŵ�� �   /     �]Y� S   2     f�� �    A     bS��    L  	   c���     A      c���    @     K����g=<                                                                                                                                                                                                                                                                                                                           BTLF 	     A       W    D      �    7      �    A         @      S   2      �   /      �   2      �   1         L  	    c   +  
   1�æ                                                                                                                                                                                                                                                                                                                                                                       OCHK    ��     0           0   REFERENCE_LIST 6     dataset        dimension                         ��             �                          #            *            Q\            �r            �(            �y            ��            ��                         �            0s              {            �u             O             �             �}             ��            ��             ^�             ��            �            R:            q            ��             ȝ�bFHDB �          l��     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 default  ?      @ 4 4�                      �?     serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                  TREE  ����������������!                       1�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    FHDB �l          ���     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������#                       R�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             FRHP               ���������      ��                           
                                                      (  �       ǃ�BTHD 	      d(�     
 
       &�eFSHD  �                             P x (        ql                   	���BTLF  W    D     �$ �   1     +�
3 �    7     ��9 "   +  	   e>� s   2     Ŵ�� D   /     �]Y�    2     f�� �   L     c���     A      c��� �    @     K�����}v                                                                                                                                                                                                                                                                                                                                            BTLF 	     A       W    D      �    7      �    @         2      D   /      s   2      �   1      �   L      "   +  	   0&��                                                                                                                                                                                                                                                                                                                                                                                    FHDB ��          �~��     _Netcdf4Coordinates                                _FillValue  ?      @ 4 4�                      � 
    is_result                                 serialised_single_element_list  ?      @ 4 4�         serialised_dicts  ?      @ 4 4�         serialised_bools    	      is_result     serialised_nones  ?      @ 4 4�         serialised_sets  ?      @ 4 4�         DIMENSION_LIST                              ��      �       _Netcdf4Dimid                                                                                                                                                                                                                                                                                                                                                                                                                                                                   TREE  ����������������#                       u�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             