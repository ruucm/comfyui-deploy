import { insertMachineSchema, machinesTable } from "@/db/schema";
import { createInsertSchema } from "drizzle-zod";

export const addMachineSchema = insertMachineSchema.pick({
  name: true,
  endpoint: true,
  type: true,
  auth_token: true,
});

export const insertCustomMachineSchema = createInsertSchema(machinesTable, {
  name: (schema) => schema.name.default("My Machine"),
  type: (schema) => schema.type.default("comfy-deploy-serverless"),
  gpu: (schema) => schema.gpu.default("T4"),
  snapshot: (schema) =>
    schema.snapshot.default({
      comfyui: "bb52934ba4e492459c5d3d01c81a8473a9962687",
      git_custom_nodes: {
        "https://github.com/BennyKok/comfyui-deploy.git": {
          hash: "503dca8fb63cbbe4d04de5f5655840a2818ade20",
          disabled: false,
        },
      },
      file_custom_nodes: [],
    }),
  models: (schema) =>
    schema.models.default([
      {
        "url": "https://civitai.com/api/download/models/113623?type=Model&format=SafeTensor&size=full&fp=fp16",
        "base": "SD 1.5",
        "name": "LEOSAM's HelloWorld XL",
        "type": "checkpoints",
        "filename": "leosamsHelloworldXL_filmGrain20.safetensors",
        "reference": "",
        "save_path": "checkpoints/SD1.5",
        "description": ""
      },
    ]),
});

export const addCustomMachineSchema = insertCustomMachineSchema.pick({
  name: true,
  type: true,
  snapshot: true,
  models: true,
  gpu: true,
});
