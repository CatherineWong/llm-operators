
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
(:action craft-wood-plank
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station WorkStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 Wood)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target WoodPlank)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 Wood))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
(:action craft-arrow
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station WeaponStation)
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 Stick)
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 Feather)
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target Arrow)
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 Stick))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 Feather))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
(:action craft-paper_1
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?ingredientinv2 - inventory ?ingredient2 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station WorkStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 SugarCane)
		(inventory-holding ?ingredientinv2 ?ingredient2)
		(object-of-type ?ingredient2 SugarCane)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Paper)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 SugarCane))
		(object-of-type ?ingredient1 Hypothetical)
		(not (inventory-holding ?ingredientinv2 ?ingredient2))
		(inventory-empty ?ingredientinv2)
		(not (object-of-type ?ingredient2 SugarCane))
		(object-of-type ?ingredient2 Hypothetical)
		)
)
(:action craft-shears1_0
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station ToolStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 IronIngot)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Shears)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 IronIngot))
		(object-of-type ?ingredient1 Hypothetical)
		)
)
(:action craft-sword_1
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?ingredientinv2 - inventory ?ingredient2 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station WeaponStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 Stick)
		(inventory-holding ?ingredientinv2 ?ingredient2)
		(object-of-type ?ingredient2 IronIngot)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Sword)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 Stick))
		(object-of-type ?ingredient1 Hypothetical)
		(not (inventory-holding ?ingredientinv2 ?ingredient2))
		(inventory-empty ?ingredientinv2)
		(not (object-of-type ?ingredient2 IronIngot))
		(object-of-type ?ingredient2 Hypothetical)
		)
)
(:action craft-bed_1
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?ingredientinv2 - inventory ?ingredient2 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station WorkStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 WoodPlank)
		(inventory-holding ?ingredientinv2 ?ingredient2)
		(object-of-type ?ingredient2 Wool)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target Bed)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 WoodPlank))
		(object-of-type ?ingredient1 Hypothetical)
		(not (inventory-holding ?ingredientinv2 ?ingredient2))
		(inventory-empty ?ingredientinv2)
		(not (object-of-type ?ingredient2 Wool))
		(object-of-type ?ingredient2 Hypothetical)
		)
)
(:action craft-iron-ingot_2
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?ingredientinv2 - inventory ?ingredient2 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station WorkStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 IronOre)
		(inventory-holding ?ingredientinv2 ?ingredient2)
		(object-of-type ?ingredient2 Coal)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target IronIngot)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 IronOre))
		(object-of-type ?ingredient1 Hypothetical)
		(not (inventory-holding ?ingredientinv2 ?ingredient2))
		(inventory-empty ?ingredientinv2)
		(not (object-of-type ?ingredient2 Coal))
		(object-of-type ?ingredient2 Hypothetical)
		)
)
(:action craft-gold-ingot_1
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?ingredientinv2 - inventory ?ingredient2 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station WorkStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 GoldOre)
		(inventory-holding ?ingredientinv2 ?ingredient2)
		(object-of-type ?ingredient2 Coal)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target GoldIngot)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 GoldOre))
		(object-of-type ?ingredient1 Hypothetical)
		(not (inventory-holding ?ingredientinv2 ?ingredient2))
		(inventory-empty ?ingredientinv2)
		(not (object-of-type ?ingredient2 Coal))
		(object-of-type ?ingredient2 Hypothetical)
		)
)
(:action cook_0
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station FoodStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 Potato)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target CookedPotato)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 Potato))
		(object-of-type ?ingredient1 Hypothetical)
		)
)
(:action cook-potato_0
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station FoodStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 Potato)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target CookedPotato)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 Potato))
		(object-of-type ?ingredient1 Hypothetical)
		)
)
(:action craft-potato2_0
        :parameters (?t - tile ?station - object ?ingredientinv1 - inventory ?ingredient1 - object ?targetinv - inventory ?target - object)

        :precondition (and 
		(agent-at ?t)
		(object-at ?station ?t)
		(object-of-type ?station FoodStation)
		(inventory-holding ?ingredientinv1 ?ingredient1)
		(object-of-type ?ingredient1 Potato)
		(inventory-empty ?targetinv)
		(object-of-type ?target Hypothetical)
		)
        :effect (and 
		(not (inventory-empty ?targetinv))
		(inventory-holding ?targetinv ?target)
		(not (object-of-type ?target Hypothetical))
		(object-of-type ?target CookedPotato)
		(not (inventory-holding ?ingredientinv1 ?ingredient1))
		(inventory-empty ?ingredientinv1)
		(not (object-of-type ?ingredient1 Potato))
		(object-of-type ?ingredient1 Hypothetical)
		)
)

    )
                