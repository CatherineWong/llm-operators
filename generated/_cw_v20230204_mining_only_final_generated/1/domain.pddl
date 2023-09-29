
    (define (domain crafting-world-v20230404-teleport)
        (:requirements :strips)
        (:types
   tile
   object
   inventory
   object-type
 )
        (:constants
  Key - object-type
  WorkStation - object-type
  Pickaxe - object-type
  IronOreVein - object-type
  IronOre - object-type
  IronIngot - object-type
  CoalOreVein - object-type
  Coal - object-type
  GoldOreVein - object-type
  GoldOre - object-type
  GoldIngot - object-type
  CobblestoneStash - object-type
  Cobblestone - object-type
  Axe - object-type
  Tree - object-type
  Wood - object-type
  WoodPlank - object-type
  Stick - object-type
  WeaponStation - object-type
  Sword - object-type
  Chicken - object-type
  Feather - object-type
  Arrow - object-type
  ToolStation - object-type
  Shears - object-type
  Sheep - object-type
  Wool - object-type
  Bed - object-type
  BedStation - object-type
  BoatStation - object-type
  Boat - object-type
  SugarCanePlant - object-type
  SugarCane - object-type
  Paper - object-type
  Furnace - object-type
  FoodStation - object-type
  Bowl - object-type
  PotatoPlant - object-type
  Potato - object-type
  CookedPotato - object-type
  BeetrootCrop - object-type
  Beetroot - object-type
  BeetrootSoup - object-type

  Hypothetical - object-type
  Trash - object-type
 )
        (:predicates
   (tile-up ?t1 - tile ?t2 - tile)     
   (tile-down ?t1 - tile ?t2 - tile)   
   (tile-left ?t1 - tile ?t2 - tile)   
   (tile-right ?t1 - tile ?t2 - tile)  

   (agent-at ?t - tile)
   (object-at ?x - object ?t - tile)
   (inventory-holding ?i - inventory ?x - object)
   (inventory-empty ?i - inventory)

   (object-of-type ?x - object ?ot - object-type)
 )
        
        (:action move-to
  :parameters (?t1 - tile ?t2 - tile)
  :precondition (and (agent-at ?t1))
  :effect (and (agent-at ?t2) (not (agent-at ?t1)))
 )
(:action pick-up
  :parameters (?i - inventory ?x - object ?t - tile)
  :precondition (and (agent-at ?t) (object-at ?x ?t) (inventory-empty ?i))
  :effect (and (inventory-holding ?i ?x) (not (object-at ?x ?t)) (not (inventory-empty ?i)))
 )
(:action place-down
  :parameters (?i - inventory ?x - object ?t - tile)
  :precondition (and (agent-at ?t) (inventory-holding ?i ?x))
  :effect (and (object-at ?x ?t) (not (inventory-holding ?i ?x)) (inventory-empty ?i))
 )
(:action mine-iron-ore
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x IronOreVein)
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool Pickaxe)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target IronOre)
  )
 )
(:action mine-wood_2
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x Tree)
		(inventory-holding ?toolinv ?tool)
		(object-of-type ?tool Axe)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Wood)
		)
)
(:action mine-wool1_0
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x Sheep)
		(inventory-holding ?toolinv ?tool)
		(object-of-type ?tool Shears)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Wool)
		)
)
(:action mine-potato_0
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x PotatoPlant)
		(inventory-holding ?toolinv ?tool)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Potato)
		)
)
(:action mine-sugar-cane_2
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x SugarCanePlant)
		(inventory-holding ?toolinv ?tool)
		(object-of-type ?tool Axe)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target SugarCane)
		)
)
(:action mine-beetroot_1
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x BeetrootCrop)
		(inventory-holding ?toolinv ?tool)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Beetroot)
		)
)
(:action mine-feather_1
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x Chicken)
		(inventory-holding ?toolinv ?tool)
		(object-of-type ?tool Sword)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Feather)
		)
)
(:action mine-cobblestone_2
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x CobblestoneStash)
		(inventory-holding ?toolinv ?tool)
		(object-of-type ?tool Pickaxe)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Cobblestone)
		)
)
(:action mine-gold-ore1_2
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x GoldOreVein)
		(inventory-holding ?toolinv ?tool)
		(object-of-type ?tool Pickaxe)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target GoldOre)
		)
)
(:action mine-coal1_0
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x CoalOreVein)
		(inventory-holding ?toolinv ?tool)
		(object-of-type ?tool Pickaxe)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Coal)
		)
)
(:action mine-beetroot1_0
        :parameters (?t - tile ?x - object ?toolinv - inventory ?tool - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?x ?t)
		(object-of-type ?x BeetrootCrop)
		(inventory-holding ?toolinv ?tool)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Beetroot)
		)
)

    )
                